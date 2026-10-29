# Third-party notices

The original benchmark code is Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Third-party models, data and installed software retain their own terms.

## Models and dataset

| Asset | Terms and attribution |
|---|---|
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/LICENSE) | Apache-2.0; Copyright 2026 Alibaba Cloud. |
| [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/LICENSE) | Apache-2.0; Copyright 2026 Alibaba Cloud. |
| [Cosmos-0.1-Tokenizer-DV8x8x8](https://huggingface.co/nvidia/Cosmos-0.1-Tokenizer-DV8x8x8/blob/bdcf9e0acc6b63074673eaed914eaaeb4894b5f3/README.md) | NVIDIA Open Model License. The pinned card links the [June 14, 2024 agreement](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf); its text is copied into `THIRD_PARTY_LICENSES.txt`. Section 2.1 addresses license updates; NVIDIA also publishes its [current agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). |
| [UCF101](https://www.crcv.ucf.edu/data/UCF101/UCF101.htm) and its [pinned ZIP mirror](https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/blob/04d4e5ca1dc93606cb58752b0c08331e598743a4/README.md) | The checked releases provide a citation but no explicit blanket license grant for the videos. Original video rights and applicable dataset terms remain with their owners. The code's Apache-2.0 license does not license these videos or the prepared RGB arrays. |

Licensed by NVIDIA Corporation under the NVIDIA Open Model License.
Built on NVIDIA Cosmos. The referenced tokenizer weights are downloaded separately.

UCF101 citation: Khurram Soomro, Amir Roshan Zamir and Mubarak Shah,
*UCF101: A Dataset of 101 Human Action Classes From Videos in The Wild*,
CRCV-TR-12-01, November 2012. [Technical report](https://arxiv.org/abs/1212.0402).

## Software

[licenses/dependencies.csv](licenses/dependencies.csv) lists **every dependency
record in `uv.lock`**, including optional and platform-specific variants. It
records the declared license, upstream/metadata URLs, distribution source and
sections in [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt).

The text file includes license, copyright and notice files supplied by the
installed Linux distributions or corresponding upstream releases. Repeated
texts are stored once with all associated components. License declarations are
reported as supplied by the publishers; a package's top-level license does not
replace the terms of its bundled components.

The main libraries are:

| Library | License |
|---|---|
| PyTorch, Torchvision | BSD-3-Clause; bundled-component notices also apply. |
| Transformers, Hugging Face Hub, Safetensors, Accelerate | Apache-2.0. |
| EasyDeL, eformer, ejkernel, JAX, jaxlib, Flax, Optax | Apache-2.0. |
| FLA core | MIT; Copyright 2023–2026 Songlin Yang, Yu Zhang, Zhiyuan Li. |
| TileLang | MIT; its TVM components carry Apache-2.0 notices. |
| causal-conv1d | BSD-3-Clause; Copyright 2023 Tri Dao. |
| Liger Kernel | BSD-2-Clause; Copyright 2024 LinkedIn Corporation. Additional bundled notices are retained. |
| NumPy | BSD-family terms, with additional bundled-library notices. |
| Pillow | MIT-CMU, with bundled-library notices. |
| PyAV | BSD-3-Clause for the wrapper; FFmpeg and codec terms are separate. |
| PyYAML | MIT. |
| Requests | Apache-2.0. |
| tqdm | MPL-2.0 and MIT; see its supplied license text. |
| libtpu | Google Cloud Platform Terms of Service, as declared by the distribution. |
| NVIDIA CUDA/runtime packages | Their supplied NVIDIA/proprietary and component terms; see the inventory and copied license files. |

PyAV wheels contain FFmpeg and codec libraries. The checked Linux PyAV 19.0.2
build reports LGPL-3.0-or-later for FFmpeg and includes x264/x265 libraries;
these have GPL terms. See [FFmpeg's licensing documentation](https://ffmpeg.org/legal.html),
the [PyAV binary build sources](https://github.com/PyAV-Org/pyav-ffmpeg), and
[x264](https://code.videolan.org/videolan/x264/-/blob/master/COPYING) /
[x265](https://bitbucket.org/multicoreware/x265_git/src/master/COPYING).
The corresponding GNU license texts are included. Retain the exact binary
build's notices and corresponding-source obligations when redistributing an
environment containing those libraries.

## Installation tools

Python 3.12 uses the PSF license and notices in its distribution. uv is offered
under Apache-2.0 or MIT. Hatchling uses MIT. Their license texts are also copied
in `THIRD_PARTY_LICENSES.txt`. The CUDA toolkit, compiler and machine image carry
the terms supplied with those installations.

This source repository does not redistribute the installed environments,
pretrained model weights or operating-system packages.
