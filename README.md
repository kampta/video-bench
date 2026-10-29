# video-bench

A GPU/TPU training benchmark for Qwen3.5-4B and Qwen3.5-27B. The task is to predict
16 future video frames from 8 history images. PyTorch and JAX implementations use
the same model weights, data, batch size and optimizer settings to compare
training speed, cost and loss.

## Benchmark setup

**Data.** UCF101, official split 1, all 101 action classes. Each clip supplies a
centered window of 24 frames at 8 FPS and 256×256 resolution. After excluding 849
short clips, the split contains 8,976 training and 3,495 held-out examples.

**Model.** The eight history images pass through Qwen's frozen ViT and merger.
The language decoder is fully fine-tuned, together with a new tied video-token
embedding and output head. During training, previous target tokens are supplied
by teacher forcing; generation uses the model's own predictions.

**Targets.** A frozen Cosmos DV8x8x8 tokenizer turns the last observed frame plus
16 future frames into three 32×32 grids. The first grid is kept for reconstruction.
The other two grids form the 2,048-token prediction target. Only history RGB
images enter Qwen's vision encoder.

**Training.** Global batch 128, seed 0, AdamW at 5e-5, 100 learning-rate warmup
updates and a 10,000-update cosine schedule. Computation uses BF16; trainable
master weights and AdamW moments use FP32. Model revisions and the full recipe
are pinned in [config.yaml](config.yaml).

**Timing.** Each run starts from the public checkpoint, performs 10 untimed
training updates, then measures 7,200 seconds. Timed steps include input loading,
frozen vision, decoder forward/backward, gradient accumulation and AdamW. Cosmos
preprocessing and fixed loss probes are outside the timed window.

## Reproduce

### Install

Use [uv](https://docs.astral.sh/uv/) from the repo root. Python and dependencies
are pinned by `.python-version` and `uv.lock`.

```sh
git clone https://github.com/kampta/video-bench.git
cd video-bench
```

On a GPU machine:

```sh
uv sync --managed-python --locked --extra cuda
source .venv/bin/activate
```

On each TPU worker:

```sh
uv sync --managed-python --locked --extra jax --extra tpu
source .venv/bin/activate
```

CUDA needs `nvcc` and `g++` for native kernels. The measured GPU profiles use H100
80 GB devices. The TPU profiles use eight physical v5p chips across two VMs.
Full 27B fine-tuning requires sharding: FP32 weights and AdamW moments alone need
about 311 GB, before gradients and activations. Provision machines separately.

### Download the prepared data

The [prepared UCF101 corpus](https://huggingface.co/datasets/kampta/video-bench-ucf101)
is public and contains the exact arrays used for the results below. Download it
once on each training machine; **no GPU, preprocessing or login is needed**:

```sh
DATA=/path/to/prepared
python -m video_bench.data --prepared --data "$DATA"
```

The download is about **21 GB**, is pinned to a fixed dataset revision, and can
resume after interruption. It includes RGB histories, target tokens, reconstruction
anchors, clip lists and data notices. Training verifies target-file hashes.
You can prefetch it on a CPU host and copy the directory to each training worker.

### Rebuild from the public videos (optional)

To create a new prepared corpus instead, run these commands once on a CUDA machine. They download the public dataset,
make an RGB preview, check codec reconstructions, and prepare the training arrays.
Use new output directories.

```sh
RAW=/path/to/ucf101
PREVIEW=/path/to/preview
DATA=/path/to/prepared

python -m video_bench.data --data "$RAW" --preview "$PREVIEW"
python -m video_bench.codec --preview "$PREVIEW" --output /path/to/reconstructions
python -m video_bench.prepare --raw "$RAW" --output "$DATA"
```

Inspect `history-future-preview.png` and `reconstructions.png`. The codec check
also verifies that the reconstruction anchor is independent of future frames.
Copy the **same prepared directory** to every training worker. Its `dataset.json`
records the accepted clips and target hashes. Fresh preprocessing also records
the encoder versions. The reference encoder environment is in `results.json`.

### Train

On every worker, set these paths to the copied data and a directory for run outputs:

```sh
DATA=/path/to/prepared
RUNS=/path/to/runs
```

Start with a six-update smoke run on one GPU:

```sh
python -m video_bench.train --backend torch --model 4b \
  --data "$DATA" --output "$RUNS/smoke" --steps 6 --timing-warmup 2
```

Then launch a full measurement using one of these commands:

```sh
# 4B: one H100
python -m video_bench.train --backend torch --model 4b \
  --data "$DATA" --output "$RUNS/4b-h100"

# 27B: eight H100s in one machine
torchrun --standalone --nproc_per_node=8 -m video_bench.train --backend torch --model 27b \
  --data "$DATA" --output "$RUNS/27b-h100"

# 4B: eight v5p chips; launch once on each TPU VM
python -m video_bench.train --backend jax --model 4b --distributed \
  --data "$DATA" --output "$RUNS/4b-v5p"

# 27B: eight v5p chips; launch once on each TPU VM
python -m video_bench.train --backend jax --model 27b --distributed \
  --data "$DATA" --output "$RUNS/27b-v5p"
```

Use the same command, data and output path on both TPU VMs. Each process uses all
local chips; `--distributed` lets JAX discover the Cloud TPU topology. Workers
must be able to communicate. Only process 0 writes metrics, so outputs may be on
worker-local disks; collect them from that worker.

The defaults below keep global batch fixed at 128 through gradient accumulation.
All use activation checkpointing. Adjust `--microbatch` when trying other hardware.

| Model | Hardware | Microbatch/device | Parallelism |
|---|---|---:|---|
| 4B | 1 H100 | 8 | Single GPU |
| 4B | 8 v5p | 16 | Data parallel |
| 27B | 8 H100 | 8 | FSDP |
| 27B | 8 v5p | 4 | FSDP |

CUDA uses eager FLA/TileLang and Liger kernels; JAX uses JIT. Every run writes
`protocol.json`, `updates.jsonl`, `probes.jsonl` and `result.json`. `--checkpoint`
saves weights after timing; JAX export currently requires a single host.
`--generate 2` adds sample GIFs on a single PyTorch GPU.

## Results

One two-hour measurement per row, using the setup above.

| Model | Fleet | $/chip/h (ref) | Updates | Ref MFU | s/update | Train@0 | Train@100 | Train@N | Val@N | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | 1 H100 | 6.538448 | 307 | 34.48% | 23.413 | 15.492391 | 9.940660 | 8.539334 | 9.025246 | 318 |
| Qwen3.5-4B | 8 v5p | 0.903271 | 971 | 29.34% | 7.413 | 15.492714 | 9.918135 | 3.590575 | 11.864931 | 982 |
| Qwen3.5-27B | 8 H100 | 2.201512 | 377 | 33.99% | 19.066 | 14.435504 | 10.822562 | 8.930257 | 10.837931 | 388 |
| Qwen3.5-27B | 8 v5p | 0.903271 | 190 | 36.91% | 37.827 | 14.436057 | 10.012901 | 9.320874 | 9.399754 | 201 |

- **Updates** counts updates completed inside the two-hour window. **N** includes
  the 10 warmups and the final update that finishes after the window.
- **Train/Val** are token cross-entropy on fixed 128-clip probes. Train probes run
  at steps 0, 100 and N; validation runs at N.
- **Ref MFU** estimates useful matmul FLOPs against dense BF16 peak: 989 TFLOP/s
  per H100 and 459 per v5p. It excludes recomputation.

Prices are GCP Spot catalog rates. The single H100 ran in `us-central1`, eight
H100s in `europe-west2`, and TPUs in `us-east5`. GPU prices include CPU and RAM;
disk and network costs are extra. Full run settings, input hashes
and step-time variation are in [results.json](results.json).

Faster runs see more training examples in two hours. The 4B TPU run makes about
14 passes over the training split and has a large train/validation gap. Losses at
different N do not compare equal training progress. BF16 training trajectories
also differ across kernels and batch geometries.

The public prepared download preserves the reference inputs exactly. Rebuilding
from videos can produce different token IDs under a different codec environment;
reuse one prepared corpus across all runs in that case.

## Code

- [data.py](video_bench/data.py), [codec.py](video_bench/codec.py), [prepare.py](video_bench/prepare.py): dataset download, inspection and tokenization.
- [workload.py](video_bench/workload.py): shared data order, image patches, positions, schedule and FLOP estimate.
- [torch_backend.py](video_bench/torch_backend.py), [jax_backend.py](video_bench/jax_backend.py): model, optimizer and device execution.
- [train.py](video_bench/train.py): common training loop, measurement and outputs.

## License

The benchmark code is [Apache-2.0](LICENSE). Models, tokenizer, dataset and
libraries have separate terms listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
UCF101 videos are not covered by the code's Apache license.

Built on NVIDIA Cosmos.
