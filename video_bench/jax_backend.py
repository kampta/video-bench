"""Public EasyDeL Qwen, sharded JAX training and FP32 AdamW state."""
from pathlib import Path

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import optax
from easydel.infra.loss_utils import LossConfig, causal_lm_loss_chunked_lm_head
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .workload import learning_rate, pixels, video_weight


class VideoModel(nnx.Module):
    def __init__(self, base, cfg, layout):
        self.base, self.layout = base, layout
        mesh = base.config.mesh
        weight = video_weight(cfg, base.config.text_config.hidden_size)
        # The BOS row makes 64,001 rows; shard the divisible hidden dimension.
        self.video = nnx.Param(jax.device_put(weight, NamedSharding(mesh, P(None, "fsdp"))))

    def __call__(self, history, labels):
        batch = len(labels)
        # Native image attention is independent per image. Vectorizing that
        # public API also keeps its interpolation/split sizes static under jit.
        patches = pixels(history, jnp).reshape(batch * 8, 256, 1536)
        def image(p):
            features, _ = self.base.get_image_features(p, ((1, 16, 16),), image_max_grid_size=16)
            return features[0]
        features = jax.vmap(image)(patches)
        prefix = jnp.broadcast_to(jnp.asarray(self.layout.prefix), (batch, len(self.layout.prefix)))
        prefix = self.base.get_input_embeddings()(prefix).astype(jnp.bfloat16)
        features = jax.lax.stop_gradient(features.reshape(batch, 512, -1))
        prefix = prefix.at[:, self.layout.image_positions].set(features)
        inputs = jnp.concatenate([jnp.full((batch, 1), self.layout.bos, jnp.int32), labels[:, :-1]], axis=1)
        x = jnp.concatenate([prefix, self.video.value[inputs].astype(jnp.bfloat16)], axis=1)
        # TPU attention tiles need a multiple of 256. The extra tokens occur
        # after every supervised position and cannot affect a causal prediction.
        pad = (-x.shape[1]) % 256
        x = jnp.pad(x, ((0, 0), (0, pad), (0, 0)))
        positions = jnp.pad(jnp.asarray(self.layout.positions), ((0, 0), (0, pad)))
        positions = jnp.broadcast_to(positions[:, None], (3, batch, x.shape[1]))
        hidden = self.base.language_model(inputs_embeds=x, position_ids=positions,
            attention_mask=jnp.ones(x.shape[:2], jnp.int32)).last_hidden_state
        hidden = hidden[:, len(self.layout.prefix):len(self.layout.prefix)+self.layout.target_length]
        return causal_lm_loss_chunked_lm_head(hidden, labels,
            lambda h: h.astype(jnp.bfloat16) @ self.video.value[:-1].astype(jnp.bfloat16).T,
            vocab_size=self.layout.bos, token_chunk_size=128,
            config=LossConfig(shift_tokens=False, compute_dtype="fp32", chunk_block_size=None)).loss


class Backend:
    def __init__(self, cfg, name, layout, microbatch, remat=True, compile_model=True, data_parallel=1):
        assert compile_model, "JAX benchmark steps use jit"
        self.cfg, self.layout = cfg, layout
        spec = cfg["models"][name]
        self.rank = jax.process_index()
        self.world = jax.device_count()
        assert data_parallel > 0 and self.world % data_parallel == 0
        self.parameter_replicas, self.parameter_shards = data_parallel, self.world // data_parallel
        self.device_kind = jax.devices()[0].device_kind
        self.microbatch = microbatch * self.world
        model = ed.AutoEasyDeLModelForImageTextToText.from_pretrained(spec["repo"], revision=spec["revision"],
            from_torch=True, token=False, dtype=jnp.bfloat16, param_dtype=jnp.float32,
            sharding_axis_dims=(data_parallel, -1, 1, 1, 1),
            config_kwargs={"attn_mechanism": "vanilla" if jax.default_backend() == "cpu" else "flash_attn2",
                           "gradient_checkpointing": "nothing_saveable" if remat else ""},
            auto_shard_model=True)
        self.mesh = model.config.mesh
        assert not model.config.vision_config.deepstack_visual_indexes
        net = VideoModel(model.model, cfg, layout)
        self.graph, self.params, self.frozen = nnx.split(net,
            lambda path, value: isinstance(value, nnx.Param) and "visual" not in path, ...)
        assert all(p.dtype == jnp.float32 for p in jax.tree.leaves(self.params)), "FP32 master weights required"
        self.trainable_parameters = sum(x.size for x in jax.tree.leaves(self.params))
        self.batch_sharding = NamedSharding(self.mesh, P(("dp", "fsdp")))
        self.tx = optax.chain(optax.clip_by_global_norm(cfg["grad_clip"]),
            optax.scale_by_adam(b1=cfg["betas"][0], b2=cfg["betas"][1], eps=cfg["eps"]),
            optax.add_decayed_weights(cfg["weight_decay"]), optax.scale(-1.))
        self.optimizer = None
        graph = self.graph

        def loss(params, frozen, history, labels):
            return nnx.merge(graph, params, frozen)(history, labels)

        self.score_fn = jax.jit(loss)
        value_grad = jax.value_and_grad(loss)

        def update(params, optimizer, frozen, history, labels, lr):
            accumulation = len(labels) // self.microbatch
            shape = (accumulation, self.microbatch)
            history = history.reshape(*shape, *history.shape[1:])
            labels = labels.reshape(*shape, *labels.shape[1:])
            micro_sharding = NamedSharding(self.mesh, P(None, ("dp", "fsdp")))
            history = jax.lax.with_sharding_constraint(history, micro_sharding)
            labels = jax.lax.with_sharding_constraint(labels, micro_sharding)

            def micro(carry, batch):
                total, grads = carry
                value, new = value_grad(params, frozen, *batch)
                return (total + value / accumulation,
                        jax.tree.map(lambda a, b: a + b / accumulation, grads, new)), None

            zeros = jax.tree.map(jnp.zeros_like, params)
            (value, grads), _ = jax.lax.scan(micro, (jnp.array(0., jnp.float32), zeros), (history, labels))
            norm = optax.global_norm(grads)
            updates, optimizer = self.tx.update(grads, optimizer, params)
            params = optax.apply_updates(params, jax.tree.map(lambda u: lr * u, updates))
            return params, optimizer, value, norm

        self.update_fn = update

    def tensors(self, history, labels):
        def put(a):
            local = np.array_split(a, jax.process_count())[self.rank]
            return jax.make_array_from_process_local_data(self.batch_sharding, local, global_shape=a.shape)
        return put(history), put(labels)

    @staticmethod
    def consensus(value, minimum=False):
        if jax.process_count() == 1:
            return value
        from jax.experimental.multihost_utils import process_allgather
        values = process_allgather(np.float32(value))
        return float(np.min(values) if minimum else np.max(values))

    def update(self, history, targets, update):
        assert len(targets) % self.microbatch == 0
        if self.optimizer is None:
            replicated = NamedSharding(self.mesh, P())
            self.optimizer = jax.tree.map(
                lambda a: jax.device_put(a, replicated) if a.ndim == 0 else a, self.tx.init(self.params))
            def layout(tree):
                return jax.tree.map(lambda a: a.sharding, tree)
            self.update_fn = jax.jit(self.update_fn, donate_argnums=(0, 1),
                out_shardings=(layout(self.params), layout(self.optimizer), None, None))
        with self.mesh:
            self.params, self.optimizer, value, norm = self.update_fn(self.params, self.optimizer, self.frozen,
                *self.tensors(history, targets), np.float32(learning_rate(self.cfg, update)))
            value, norm = jax.block_until_ready((value, norm))
        return {"loss": float(value), "grad_norm": float(norm)}

    def score(self, history, targets):
        assert len(targets) % self.microbatch == 0
        total = 0.
        with self.mesh:
            for start in range(0, len(targets), self.microbatch):
                value = self.score_fn(self.params, self.frozen,
                    *self.tensors(history[start:start+self.microbatch], targets[start:start+self.microbatch]))
                total += float(value.block_until_ready()) * self.microbatch
        return total / len(targets)

    def save(self, path):
        net = nnx.merge(self.graph, self.params, self.frozen)
        path = Path(path)
        net.base.save_pretrained(path / "base")
        if self.rank == 0:
            np.save(path / "video.npy", np.asarray(net.video.value))
