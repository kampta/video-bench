"""PyTorch Qwen with native frozen vision, FLA, Liger and FP32 AdamW state."""
import os

import numpy as np
import torch
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss, apply_liger_kernel_to_qwen3_5
from torch import nn
from transformers import Qwen3_5ForConditionalGeneration

from .workload import learning_rate, pixels, video_weight


class VideoModel(nn.Module):
    def __init__(self, cfg, name, layout, device, remat, sharded=False):
        super().__init__()
        apply_liger_kernel_to_qwen3_5(fused_linear_cross_entropy=False)
        spec = cfg["models"][name]
        wrapper, loaded = Qwen3_5ForConditionalGeneration.from_pretrained(spec["repo"], revision=spec["revision"],
            token=False, dtype=torch.float32, device_map={"": "cpu" if sharded else device}, attn_implementation="sdpa",
            output_loading_info=True)
        assert not loaded["missing_keys"], f"Public checkpoint has {len(loaded['missing_keys'])} missing weights"
        self.base = wrapper.model
        self.base.visual.requires_grad_(False).to(dtype=torch.bfloat16).eval()
        if remat:
            self.base.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        self.video = nn.Parameter(torch.from_numpy(video_weight(cfg, self.base.config.text_config.hidden_size)).to(device))
        self.layout = layout
        self.register_buffer("prefix", torch.tensor(layout.prefix, device=device), persistent=False)
        self.register_buffer("positions", torch.tensor(layout.positions, device=device), persistent=False)
        self.register_buffer("image_positions", torch.tensor(layout.image_positions, device=device), persistent=False)
        self.ce = LigerFusedLinearCrossEntropyLoss(accum_dtype=torch.float32)

    def embeddings(self, history, video_ids):
        batch = len(video_ids)
        grid = torch.tensor(np.tile(self.layout.grid, (batch, 1)))
        with torch.no_grad():
            features = torch.cat(self.base.get_image_features(pixels(history, torch), grid).pooler_output)
        prefix = self.base.get_input_embeddings()(self.prefix)[None].expand(batch, -1, -1).to(torch.bfloat16).clone()
        prefix[:, self.image_positions] = features.reshape(batch, 512, -1)
        video = nn.functional.embedding(video_ids, self.video).to(torch.bfloat16)
        return torch.cat([prefix, video], dim=1)

    def forward(self, history, video_ids, labels):
        x = self.embeddings(history, video_ids)
        output = self.base.language_model(inputs_embeds=x,
            position_ids=self.positions[:, None].expand(-1, len(labels), -1), use_cache=False)
        hidden = output.last_hidden_state[:, len(self.layout.prefix):].reshape(-1, self.video.shape[1])
        weight = self.video[:-1] if self.training else self.video[:-1].detach()
        return self.ce(weight, hidden, labels.reshape(-1))


class Backend:
    def __init__(self, cfg, name, layout, microbatch, remat=True, compile_model=False):
        assert not compile_model, "CUDA uses eager FLA/TileLang kernels; omit --compile"
        os.environ.setdefault("FLA_TILELANG", "1")
        self.cfg, self.layout, self.microbatch = cfg, layout, microbatch
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.parameter_replicas, self.parameter_shards = 1, self.world
        device = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(device)
        if torch.cuda.get_device_capability(device)[0] == 9:
            from fla.backends import TileLangBackend
            assert TileLangBackend.is_available(), "H100 needs TileLang, nvcc and g++; check the CUDA toolchain"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(cfg["seed"])
        if self.world > 1:
            torch.distributed.init_process_group("nccl")
        self.model = VideoModel(cfg, name, layout, device, remat, self.world > 1)
        if self.world > 1:
            from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
            policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            for layer in self.model.base.language_model.layers:
                fully_shard(layer, mp_policy=policy)
            fully_shard(self.model, mp_policy=policy)
        self.decoder = self.model.base.language_model
        self.optimizer = None
        self.device = device
        self.device_kind = torch.cuda.get_device_name(device)
        assert all(p.dtype == torch.float32 for p in self.model.parameters() if p.requires_grad), "FP32 masters required"
        self.trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def tensors(self, history, targets):
        return [torch.from_numpy(a).to(self.device, non_blocking=True) for a in
                (history, self.layout.video_inputs(targets), targets.astype(np.int64))]

    def consensus(self, value, minimum=False):
        if self.world == 1:
            return value
        x = torch.tensor(value, device=self.device, dtype=torch.float64)
        op = torch.distributed.ReduceOp.MIN if minimum else torch.distributed.ReduceOp.MAX
        torch.distributed.all_reduce(x, op=op)
        return float(x)

    def update(self, history, targets, update):
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                lr=self.cfg["lr"], betas=tuple(self.cfg["betas"]), eps=self.cfg["eps"],
                weight_decay=self.cfg["weight_decay"], fused=True)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate(self.cfg, update)
        self.optimizer.zero_grad(set_to_none=True)
        history, targets = history[self.rank::self.world], targets[self.rank::self.world]
        assert len(targets) % self.microbatch == 0
        accumulation = len(targets) // self.microbatch
        total = torch.zeros((), device=self.device)
        self.model.train()
        self.model.base.visual.eval()
        for i in range(accumulation):
            # Reduce-scatter each microbatch to keep FP32 gradients sharded.
            start = i * self.microbatch
            batch = self.tensors(history[start:start+self.microbatch], targets[start:start+self.microbatch])
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                loss = self.model(*batch) / accumulation
            loss.backward()
            total += loss.detach()
        norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.world > 1:
            torch.distributed.all_reduce(total)
            total /= self.world
        torch.cuda.synchronize()
        return {"loss": float(total), "grad_norm": float(norm),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated()}

    @torch.no_grad()
    def score(self, history, targets):
        self.model.eval()
        history, targets = history[self.rank::self.world], targets[self.rank::self.world]
        total = 0.
        for start in range(0, len(targets), self.microbatch):
            batch = self.tensors(history[start:start+self.microbatch], targets[start:start+self.microbatch])
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                total += float(self.model(*batch)) * len(batch[2])
        if self.world > 1:
            total = torch.tensor(total, device=self.device, dtype=torch.float64)
            torch.distributed.all_reduce(total)
        return float(total) / (len(targets) * self.world)

    @torch.no_grad()
    def generate(self, history):
        assert self.world == 1, "Generate with a loaded, unsharded export"
        self.model.eval()
        self.optimizer = None  # Sampling does not need the FP32 moments.
        image = torch.from_numpy(history).to(self.device)
        ids = torch.full((len(history), 1), self.layout.bos, device=self.device, dtype=torch.long)
        result, cache = [], None
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
            x = self.model.embeddings(image, ids)
            for i in range(self.layout.target_length):
                start = 0 if i == 0 else len(self.layout.prefix) + i
                end = len(self.layout.prefix) + i + 1
                out = self.decoder(inputs_embeds=x, past_key_values=cache, use_cache=True,
                    position_ids=self.model.positions[:, None, start:end])
                cache = out.past_key_values
                logits = nn.functional.linear(out.last_hidden_state[:, -1], self.model.video[:-1])
                ids = logits.argmax(-1)
                result.append(ids)
                x = nn.functional.embedding(ids[:, None], self.model.video).to(torch.bfloat16)
        return torch.stack(result, dim=1).cpu().numpy()

    def save(self, path):
        from safetensors.torch import save_file
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
        state = get_model_state_dict(self.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if self.rank == 0:
            save_file({k: v.contiguous() for k, v in state.items()}, str(path))
