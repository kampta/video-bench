"""The data order, image patches, multimodal positions and targets shared by both backends."""
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model


class Layout:
    def __init__(self, cfg, model):
        spec = cfg["models"][model]
        p = AutoProcessor.from_pretrained(spec["repo"], revision=spec["revision"], token=False)
        mc = AutoConfig.from_pretrained(spec["repo"], revision=spec["revision"], token=False)
        self.model_config = mc
        ds = cfg["dataset"]
        assert ds["resolution"] == 256 and ds["history_frames"] == 8
        assert (p.image_processor.patch_size, p.image_processor.temporal_patch_size,
                p.image_processor.merge_size) == (16, 2, 2)
        assert list(p.image_processor.image_mean) == list(p.image_processor.image_std) == [0.5] * 3
        content = []
        for i in range(ds["history_frames"]):
            content.extend([{"type": "text", "text": f"t={(i-7)/ds['fps']:+.3f}s"}, {"type": "image"}])
        content.append({"type": "text", "text": "Predict the next sixteen video frames."})
        prompt = p.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                      add_generation_prompt=True)
        batch = p(text=[prompt], images=[np.zeros((256, 256, 3), np.uint8)] * 8, return_tensors="np")
        self.prefix = batch["input_ids"][0]
        self.image_positions = np.flatnonzero(self.prefix == mc.image_token_id)
        self.grid = batch["image_grid_thw"].astype(np.int32)
        self.target_length = 2048
        self.bos = cfg["codec"]["vocabulary"]
        stop = len(self.prefix) + self.target_length
        self.length = cfg["context"] or stop
        assert stop <= self.length and len(self.image_positions) == 512
        ids = np.pad(self.prefix, (0, self.length - len(self.prefix)))
        types = np.pad(batch["mm_token_type_ids"][0], (0, self.length - len(self.prefix)))
        self.mask = (np.arange(self.length) < stop).astype(np.int32)
        helper = SimpleNamespace(config=mc)
        helper.get_vision_position_ids = MethodType(Qwen3_5Model.get_vision_position_ids, helper)
        positions, _ = Qwen3_5Model.get_rope_index(helper, torch.tensor(ids[None]), torch.tensor(types[None]),
                        image_grid_thw=torch.tensor(self.grid), attention_mask=torch.tensor(self.mask[None]))
        self.positions = positions.numpy()[:, 0].astype(np.int32)

    def video_inputs(self, targets):
        assert targets.shape[1] == self.target_length
        return np.concatenate([np.full((len(targets), 1), self.bos, np.int32), targets[:, :-1]], axis=1)


def pixels(history, xp=np):
    """Native Qwen image patches for already resized 256px uint8 RGB images."""
    assert history.dtype == xp.uint8 and history.shape[1:] == (8, 256, 256, 3)
    # Qwen's fast image processor performs fused rescale/normalization in FP32.
    x = (xp.asarray(history, dtype=xp.float32) - 127.5) / 127.5
    x = xp.moveaxis(x, 4, 2).reshape(-1, 3, 256, 256)
    x = xp.tile(x[:, None], (1, 2, 1, 1, 1)).reshape(-1, 1, 2, 3, 8, 2, 16, 8, 2, 16)
    return xp.moveaxis(x, (0, 1, 4, 7, 5, 8, 3, 2, 6, 9), tuple(range(10))).reshape(-1, 1536)


def video_weight(cfg, hidden):
    return np.random.default_rng(cfg["seed"]).normal(0, 0.02,
        (cfg["codec"]["vocabulary"] + 1, hidden)).astype(np.float32)


def learning_rate(cfg, update):
    if update < cfg["warmup_updates"]:
        return cfg["lr"] * (update + 1) / cfg["warmup_updates"]
    t = min(1., (update - cfg["warmup_updates"]) / (cfg["schedule_updates"] - cfg["warmup_updates"]))
    return cfg["lr"] * (cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * (1 + np.cos(np.pi * t)) / 2)


class Dataset:
    def __init__(self, root, split, seed):
        root = Path(root)
        self.manifest = json.loads((root / "dataset.json").read_text())
        self.n = self.manifest[split]["count"]
        with (root / split / "targets.npy").open("rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == self.manifest["target_hashes"][split], \
                "Prepared targets changed; prepare or copy the data again"
        self.history = np.load(root / split / "history.npy", mmap_mode="r")
        self.targets = np.load(root / split / "targets.npy", mmap_mode="r")
        assert 0 < self.n <= min(len(self.history), len(self.targets)), "Invalid prepared split count"
        assert self.history.shape[1:] == (8, 256, 256, 3) and self.history.dtype == np.uint8
        assert self.targets.shape[1:] == (2048,) and self.targets.dtype == np.uint16
        self.order = np.random.default_rng(seed).permutation(self.n)

    def batch(self, cursor, count):
        idx = self.order[(cursor + np.arange(count)) % self.n]
        return self.history[idx], self.targets[idx].astype(np.int32)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def reference_flops(layout):
    """Approximate useful forward/backward matmul FLOPs; excludes checkpoint recomputation."""
    c, v = layout.model_config.text_config, layout.model_config.vision_config
    h, length = c.hidden_size, len(layout.prefix) + layout.target_length
    full = c.layer_types.count("full_attention")
    linear = c.num_hidden_layers - full
    q, k = c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim
    dense = 3 * c.num_hidden_layers * h * c.intermediate_size
    dense += full * h * (3 * q + 2 * k)
    dense += linear * h * (2 * c.linear_num_key_heads * c.linear_key_head_dim
                          + 3 * c.linear_num_value_heads * c.linear_value_head_dim + 2 * c.linear_num_value_heads)
    decoder = 6 * dense * length + 6 * layout.bos * h * layout.target_length
    decoder += 6 * full * length * (length + 1) * q  # Causal QK/AV and their gradients.
    decoder += 22 * linear * length * c.linear_num_value_heads * c.linear_key_head_dim * c.linear_value_head_dim
    patches, merge = (256 // v.patch_size) ** 2, v.spatial_merge_size ** 2
    vision_dense = v.depth * (4 * v.hidden_size**2 + 2 * v.hidden_size * v.intermediate_size)
    merger = (merge * v.hidden_size)**2 + merge * v.hidden_size * h
    stem = 3 * v.temporal_patch_size * v.patch_size**2 * v.hidden_size
    vision = 2 * 8 * patches * (vision_dense + stem + merger / merge)
    vision += 4 * 8 * v.depth * patches**2 * v.hidden_size
    return decoder + vision
