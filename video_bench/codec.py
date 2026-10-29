"""Encode and reconstruct video with the frozen Cosmos DV8x8x8 tokenizer."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

from .data import config


class Codec:
    def __init__(self, cfg, device="cuda"):
        spec = cfg["codec"]
        self.device = device
        for name in ("encoder", "decoder"):
            path = hf_hub_download(spec["repo"], name + ".jit", revision=spec["revision"], token=False)
            setattr(self, name, torch.jit.load(path, map_location=device).eval())

    @torch.inference_mode()
    def encode(self, frames):
        x = torch.as_tensor(frames, device=self.device).permute(3, 0, 1, 2)[None]
        x = (x.float() / 127.5 - 1).to(next(self.encoder.parameters()).dtype)
        # Keep BF16 quantization independent of TorchScript profiling state.
        with torch.jit.optimized_execution(False):
            indices, _, _ = self.encoder(x)
        return indices[0].cpu().numpy().astype(np.uint16)

    @torch.inference_mode()
    def decode(self, indices):
        with torch.jit.optimized_execution(False):
            x = self.decoder(torch.as_tensor(indices.astype(np.int32), device=self.device)[None])
        return ((x[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()

    def anchor(self, frame):
        # Match the 17-frame convolution shape without using future pixels.
        observed = np.zeros((17, *frame.shape), dtype=frame.dtype)
        observed[0] = frame
        return self.encode(observed)[:1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--preview", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(8)
    cfg = config(args.config)
    assert torch.cuda.is_available(), "The published BF16 Cosmos tokenizer requires CUDA"
    model = Codec(cfg, args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    files = sorted(args.preview.glob("*.npy"))
    sheet = Image.new("RGB", (8 * 128, len(files) * 152), "white")
    draw = ImageDraw.Draw(sheet)
    for row, path in enumerate(files):
        frames = np.load(path)[cfg["dataset"]["history_frames"] - 1:]
        indices = model.encode(frames)
        anchor = model.anchor(frames[0])
        assert indices.shape == (3, 32, 32), indices.shape
        assert np.array_equal(indices[:1], anchor), "Codec's observed anchor depends on future frames"
        assert indices.max() < cfg["codec"]["vocabulary"], "Codec ID exceeds video vocabulary"
        reconstruction = model.decode(indices)
        assert reconstruction.shape == frames.shape
        np.savez(args.output / (path.stem + ".npz"), indices=indices, reconstruction=reconstruction)
        # Four future moments: original immediately above its codec reconstruction.
        for col, i in enumerate((1, 6, 11, 16)):
            for delta, video in ((0, frames), (1, reconstruction)):
                sheet.paste(Image.fromarray(video[i]).resize((128, 128)), ((col * 2 + delta) * 128, row * 152 + 24))
        draw.text((0, row * 152), path.stem + ": future RGB / reconstruction pairs", fill="black")
        record = {"clip": path.stem, "codec_dtype": "bfloat16", "latent_shape": list(indices.shape), "predicted_tokens": int(indices[1:].size),
                  "anchor_causal_exact": True, "future_pixel_MAE": float(np.abs(frames[1:].astype(float) - reconstruction[1:]).mean())}
        records.append(record)
        print(json.dumps(record), flush=True)
    sheet.save(args.output / "reconstructions.png")
    (args.output / "reconstruction.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
