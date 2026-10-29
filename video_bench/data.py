"""Download prepared UCF101 inputs or inspect the original videos."""
import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import av
import numpy as np
import yaml
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image, ImageDraw


def config(path="config.yaml"):
    return yaml.safe_load(Path(path).read_text())


def download(cfg, root):
    ds = cfg["dataset"]
    return [Path(hf_hub_download(ds["repo"], ds[key], repo_type="dataset", revision=ds["revision"],
                                token=False, local_dir=root)) for key in ("archive", "split_archive")]


def split_lists(annotations, split):
    with zipfile.ZipFile(annotations) as z:
        values = {name: [line.split()[0] for line in z.read(f"ucfTrainTestlist/{prefix}{split:02d}.txt")
                        .decode().splitlines()] for name, prefix in (("train", "trainlist"), ("val", "testlist"))}
    groups = {name: {re.sub(r"_c\d+\.avi$", "", p) for p in paths} for name, paths in values.items()}
    assert not groups["train"] & groups["val"], "Official train/test groups overlap"
    return values


def clip(archive, member, cfg):
    ds = cfg["dataset"]
    count = ds["history_frames"] + ds["future_frames"]
    with av.open(io.BytesIO(archive.read("UCF-101/" + member))) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base)
        span = (count - 1) / ds["fps"]
        if duration < span:
            raise ValueError(f"Clip duration {duration:.2f}s is below required {span:.2f}s")
        times = (duration - span) / 2 + np.arange(count) / ds["fps"]
        frames = []
        for frame in container.decode(video=0):
            if frame.time is None or frame.time < times[len(frames)]:
                continue
            image = frame.to_image()
            side = min(image.size)
            left, top = ((n - side) // 2 for n in image.size)
            image = image.crop((left, top, left + side, top + side)).resize(
                (ds["resolution"], ds["resolution"]), Image.Resampling.BICUBIC)
            frames.append(np.asarray(image))
            if len(frames) == count:
                return np.stack(frames)
    raise ValueError("Video ended before the requested sampled window")


def preview(cfg, root, output):
    archive, annotations = download(cfg, root)
    splits = split_lists(annotations, cfg["dataset"]["split"])
    output.mkdir(parents=True, exist_ok=False)
    examples, rejected = [], []
    with zipfile.ZipFile(archive) as z:
        for split, paths in splits.items():
            classes = set()
            for path in paths:
                if path.split("/")[0] in classes:
                    continue
                try:
                    frames = clip(z, path, cfg)
                except ValueError as error:
                    rejected.append({"clip": path, "reason": str(error)})
                    continue
                classes.add(path.split("/")[0])
                name = f"{split}-{len(classes)}"
                np.save(output / f"{name}.npy", frames)
                examples.append({"name": name, "split": split, "clip": path,
                                 "history_frames": cfg["dataset"]["history_frames"],
                                 "future_frames": cfg["dataset"]["future_frames"]})
                if len(classes) == 2:
                    break
    sheet = Image.new("RGB", (12 * 100, len(examples) * 132), "white")
    draw = ImageDraw.Draw(sheet)
    for row, example in enumerate(examples):
        frames = np.load(output / (example["name"] + ".npy"))
        draw.text((0, row * 132), f"{example['split']}: {example['clip']} | 8 history, then future samples", fill="black")
        indices = [*range(8), 8, 13, 18, 23]
        for col, i in enumerate(indices):
            sheet.paste(Image.fromarray(frames[i]).resize((96, 96)), (col * 100, row * 132 + 18))
            draw.text((col * 100, row * 132 + 115), f"{(i-7)/cfg['dataset']['fps']:+.3f}s", fill="black")
    sheet.save(output / "history-future-preview.png")
    (output / "preview.json").write_text(json.dumps({"config": cfg["dataset"], "examples": examples,
         "official_split_counts": {s: len(v) for s, v in splits.items()}, "rejected_preview_candidates": rejected,
         "history_future_overlap": False, "train_validation_group_overlap": False,
         "tokenizer_reconstruction_pending": True}, indent=2) + "\n")
    print(json.dumps({"preview": str(output / "history-future-preview.png"), "examples": len(examples)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", type=Path, required=True, help="Destination for downloaded data")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", type=Path, help="Download raw videos and write an RGB preview")
    mode.add_argument("--prepared", action="store_true", help="Download the prepared benchmark corpus; no GPU needed")
    args = parser.parse_args()
    if args.prepared:
        snapshot_download("kampta/video-bench-ucf101", repo_type="dataset",
                          revision="1beef16d6a9132e21565a891edffacb01de92a44",
                          token=False, local_dir=args.data)
        print(json.dumps({"prepared": str(args.data)}))
    else:
        preview(config(args.config), args.data, args.preview)


if __name__ == "__main__":
    main()
