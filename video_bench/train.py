"""Train and measure Qwen video prediction with PyTorch or JAX."""
import argparse
import datetime as dt
import importlib
import json
import math
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .data import config
from .workload import Dataset, Layout, fingerprint, reference_flops


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--backend", choices=("torch", "jax"), required=True)
    p.add_argument("--model", choices=("4b", "27b"), required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--microbatch", type=int, help="Resident examples per device; 4B/H100 defaults to 8")
    p.add_argument("--remat", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                   help="JIT is required for JAX; CUDA uses eager FLA/TileLang kernels")
    p.add_argument("--steps", type=int, help="Short functional/tuning run; omitted for the full window")
    p.add_argument("--timing-warmup", type=int)
    p.add_argument("--checkpoint", action="store_true")
    p.add_argument("--generate", type=int, default=0, help="PyTorch qualitative samples after training")
    p.add_argument("--distributed", action="store_true", help="Initialize a multi-host JAX runtime")
    p.add_argument("--data-parallel", type=int, help="JAX parameter replicas; remaining devices use FSDP")
    p.add_argument("--stop-at", help="Optional UTC deadline; does not shorten an admitted measurement")
    p.add_argument("--source-id", help="Commit or source-archive hash for this run")
    p.add_argument("--peak-tflops", type=float, help="Dense BF16 peak per device, for reference MFU")
    args = p.parse_args()
    if args.compile is None:
        args.compile = args.backend == "jax"
    cfg = config(args.config)
    assert args.steps is None or args.steps >= 0
    assert args.generate == 0 or args.backend == "torch", "Sampling currently uses the PyTorch export"
    v5p_8 = False
    h100_8 = False
    if args.backend == "jax":
        import jax
        if args.distributed:
            jax.distributed.initialize()
        v5p_8 = jax.device_count() == 8 and jax.devices()[0].device_kind == "TPU v5"
        if args.steps is None:
            assert jax.default_backend() == "tpu", "Full JAX measurements require native TPU devices"
        if args.checkpoint:
            assert jax.process_count() == 1, "Use a single-host JAX export; worker-local paths are not shared storage"
    elif args.model == "27b" and int(os.environ.get("WORLD_SIZE", "1")) == 8:
        import torch
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        h100_8 = torch.cuda.device_count() == 8 and torch.cuda.get_device_name() == "NVIDIA H100 80GB HBM3"
    if args.data_parallel is None:
        args.data_parallel = 8 if v5p_8 and args.model == "4b" else 1
    if args.microbatch is None:
        args.microbatch = (16 if args.model == "4b" else 4) if v5p_8 else \
            (8 if args.backend == "torch" and (args.model == "4b" or h100_8) else 1)
    assert args.data_parallel == 1 or args.backend == "jax", "This replica setting is for JAX"
    world = jax.device_count() if args.backend == "jax" else int(os.environ.get("WORLD_SIZE", "1"))
    assert args.microbatch > 0 and cfg["global_batch"] % (args.microbatch * world) == 0, \
        "Global batch must be divisible by microbatch times device count"
    layout = Layout(cfg, args.model)
    train, val = [Dataset(args.data, s, cfg["seed"]) for s in ("train", "val")]
    assert (train.manifest["dataset"], train.manifest["codec"]) == (cfg["dataset"], cfg["codec"]), \
        "Prepare the public data with this pinned config before training"
    if args.steps is None:
        assert not train.manifest["debug_subset"], "Full benchmarks require the complete prepared public split"
        assert cfg["measured_seconds"] == 7200, "The measured comparison uses exactly a two-hour target"
    options = {"data_parallel": args.data_parallel} if args.backend == "jax" else {}
    backend = importlib.import_module(f"video_bench.{args.backend}_backend").Backend(
        cfg, args.model, layout, args.microbatch, args.remat, args.compile, **options)
    if args.peak_tflops is None:
        args.peak_tflops = {"NVIDIA H100 80GB HBM3": 989, "TPU v5": 459,
                           "TPU v5 lite": 197, "TPU v6e": 918}.get(backend.device_kind)
    assert not args.generate or backend.world == 1, "Sample from an unsharded export"
    if backend.rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"config": cfg, "model": args.model, "dataset_sha256": fingerprint(train.manifest),
                "logical_sequence_length": layout.length, "history_images": 8, "future_tokens": 2048,
                "source_id": args.source_id,
                "train_probe_indices": train.order[:cfg["probe_examples"]].tolist(),
                "val_probe_indices": val.order[:cfg["probe_examples"]].tolist()}
    protocol_sha = fingerprint(protocol)
    if backend.rank == 0:
        (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    probes, updates = [], []
    n, measured = 0, 0.
    warmup = cfg["timing_warmup_updates"] if args.timing_warmup is None else args.timing_warmup
    wall_started = time.monotonic()

    def emit(filename, value):
        if backend.rank == 0:
            with (args.output / filename).open("a") as f:
                f.write(json.dumps(value) + "\n")
            print(json.dumps(value), flush=True)

    def probe(split, dataset):
        loss = backend.score(*dataset.batch(0, cfg["probe_examples"]))
        assert math.isfinite(loss), "Non-finite fixed-probe NLL"
        value = {"kind": "fixed_probe", "split": split, "update": n,
                 "examples": cfg["probe_examples"], "loss": loss, "protocol_sha256": protocol_sha}
        probes.append(value)
        emit("probes.jsonl", value)

    probe("train", train)
    while (n < args.steps if args.steps is not None else measured < cfg["measured_seconds"]):
        if args.stop_at:
            remaining = (dt.datetime.fromisoformat(args.stop_at) - dt.datetime.now(dt.UTC)).total_seconds()
            remaining = backend.consensus(remaining, minimum=True)
            required = 0 if args.steps is not None else cfg["measured_seconds"] - measured
            assert remaining > required + 120, "Insufficient remaining time for the full measurement and final probes"
        started = time.monotonic()
        result = backend.update(*train.batch(n * cfg["global_batch"], cfg["global_batch"]), n)
        seconds = backend.consensus(time.monotonic() - started)
        assert math.isfinite(result["loss"]) and math.isfinite(result["grad_norm"]), "Non-finite training update"
        n += 1
        timed = n > warmup
        in_window = timed and measured + seconds <= cfg["measured_seconds"]
        if timed:
            measured += seconds
        row = {"kind": "update", "N": n, "cursor": n * cfg["global_batch"], "seconds": seconds,
               "timed": timed, "in_window": in_window, "measured_seconds": measured, **result}
        updates.append(row)
        emit("updates.jsonl", row)
        if n == 100:
            probe("train", train)
    if n != 0 and n != 100:
        probe("train", train)
    probe("val", val)
    timed_rows = [u for u in updates if u["timed"]]
    result = {"complete": True, "full_two_hour_measurement": args.steps is None and measured >= 7200,
        "backend": args.backend, "model": args.model, "devices": backend.world,
        "hardware": backend.device_kind,
        "parameter_replicas": backend.parameter_replicas, "parameter_shards": backend.parameter_shards,
        "microbatch_per_device": args.microbatch, "global_batch": cfg["global_batch"],
        "rematerialization": args.remat, "compiled": args.compile,
        "trainable_parameters": backend.trainable_parameters, "N": n,
        "timing_warmup_updates": min(n, warmup), "updates": sum(u["in_window"] for u in updates),
        "timed_updates": len(timed_rows), "late_updates": sum(not u["in_window"] for u in timed_rows),
        "measured_seconds": measured, "seconds_per_update": measured / len(timed_rows) if timed_rows else None,
        "wall_seconds_excluding_save_and_generation": time.monotonic() - wall_started,
        "step_clock": "shared maximum host duration",
        "probes": probes, "protocol_sha256": protocol_sha, "python": platform.python_version(),
        "master_parameters_and_optimizer": "float32", "computation": "bfloat16",
        "vision_frozen_and_included_in_step": True, "codec_preprocessing_included_in_step": False}
    result["reference_flops_per_update"] = reference_flops(layout) * cfg["global_batch"]
    result["reference_MFU"] = (result["reference_flops_per_update"] / result["seconds_per_update"]
        / (args.peak_tflops * 1e12 * backend.world)) if args.peak_tflops and timed_rows else None
    if backend.rank == 0:
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.checkpoint:
        backend.save(args.output / ("model.safetensors" if args.backend == "torch" else "model"))
    if args.generate and backend.rank == 0:
        import numpy as np

        from .codec import Codec
        codec = Codec(cfg)
        history, _ = val.batch(0, args.generate)
        codes = backend.generate(history).astype(np.uint16).reshape(-1, 2, 32, 32)
        anchors = np.load(args.data / "val/anchors.npy", mmap_mode="r")[val.order[:args.generate]]
        for i, (anchor, future) in enumerate(zip(anchors, codes)):
            frames = codec.decode(np.concatenate([anchor[None], future]))
            from PIL import Image
            images = [Image.fromarray(f) for f in frames]
            images[0].save(args.output / f"sample-{i}.gif", save_all=True, append_images=images[1:], duration=125, loop=0)
    if backend.rank == 0:
        print(json.dumps(result), flush=True)
    if args.backend == "jax":
        import jax
        from jax.experimental.multihost_utils import sync_global_devices
        sync_global_devices("training_complete")
        if jax.process_count() > 1:
            jax.distributed.shutdown()


if __name__ == "__main__":
    main()
