"""Prepare public RGB histories and frozen Cosmos targets once, outside training timing."""
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from .codec import Codec
from .data import clip, config, download, split_lists


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, help="Debug subset per split; omitted for benchmark data")
    args = p.parse_args()
    assert torch.cuda.is_available(), "Public Cosmos preprocessing needs CUDA"
    torch.set_num_threads(8)
    cfg = config(args.config)
    archive, annotations = download(cfg, args.raw)
    lists = split_lists(annotations, cfg["dataset"]["split"])
    args.output.mkdir(parents=True, exist_ok=False)
    codec = Codec(cfg)
    result = {"dataset": cfg["dataset"], "codec": cfg["codec"], "codec_dtype": "bfloat16",
              "producer": {"torch": torch.__version__, "cuda": torch.version.cuda},
              "debug_subset": args.limit is not None, "source_archive_bytes": archive.stat().st_size}
    started = time.monotonic()
    with zipfile.ZipFile(archive) as z:
        for split, paths in lists.items():
            paths = paths[:args.limit] if args.limit else paths
            root = args.output / split
            root.mkdir()
            histories = open_memmap(root / "history.npy", mode="w+", dtype=np.uint8,
                                    shape=(len(paths), 8, 256, 256, 3))
            targets = open_memmap(root / "targets.npy", mode="w+", dtype=np.uint16, shape=(len(paths), 2048))
            anchors = open_memmap(root / "anchors.npy", mode="w+", dtype=np.uint16, shape=(len(paths), 32, 32))
            accepted, rejected = [], []
            for path in paths:
                try:
                    frames = clip(z, path, cfg)
                except ValueError as error:
                    rejected.append({"clip": path, "reason": str(error)})
                    continue
                codes = codec.encode(frames[7:])
                assert codes.shape == (3, 32, 32) and codes.max() < cfg["codec"]["vocabulary"]
                i = len(accepted)
                if i == 0:
                    assert np.array_equal(codes[:1], codec.anchor(frames[7])), "Future-dependent anchor"
                histories[i], targets[i], anchors[i] = frames[:8], codes[1:].reshape(-1), codes[0]
                accepted.append(path)
                if (i + 1) % 100 == 0:
                    print(json.dumps({"split": split, "clips": i + 1, "elapsed_seconds": time.monotonic()-started}), flush=True)
            histories.flush()
            targets.flush()
            anchors.flush()
            result[split] = {"count": len(accepted), "clips": accepted, "rejected": rejected}
    result["prepared_seconds"] = time.monotonic() - started
    result["target_hashes"] = {s: hashlib.sha256((args.output/s/"targets.npy").read_bytes()).hexdigest()
                               for s in lists}
    (args.output / "dataset.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "counts": {s: result[s]["count"] for s in lists},
                      "prepared_seconds": result["prepared_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
