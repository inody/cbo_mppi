# Third-party software notices

This repository contains original CBO-MPPI source code and declares external
Python packages as dependencies. It does not vendor their source code or binary
artifacts. Each dependency remains governed by its own license.

## Direct runtime dependencies

| Package | Version pinned by this repository | License |
|---|---:|---|
| JAX | 0.8.3 | Apache License 2.0 |
| jaxlib | 0.8.3 | Apache License 2.0 |
| NumPy | 2.4.6 | BSD-3-Clause and other permissive licenses for bundled components |
| Matplotlib | 3.11.0 | Matplotlib License (PSF-compatible permissive license) |

## Optional CUDA installation

The `cuda` optional dependency installs the JAX CUDA plugin and NVIDIA CUDA
runtime packages from their upstream distribution channels. The JAX CUDA
plugin is Apache-2.0. NVIDIA CUDA, cuBLAS, cuDNN, NCCL, and related binary
packages are distributed under NVIDIA's applicable proprietary license terms.
They are not included in this source repository.

Anyone redistributing a container, environment archive, wheel bundle, or other
binary distribution containing NVIDIA components must review and comply with
the corresponding NVIDIA terms. Merely publishing this source repository and
its lock file does not redistribute those binaries.

## Transitive dependencies

The locked environment also uses permissively licensed packages including
SciPy (BSD), Pillow (MIT-CMU), ContourPy (BSD-3-Clause), Cycler (BSD),
FontTools (MIT), KiwiSolver (BSD), ml_dtypes (Apache-2.0), opt_einsum (MIT),
Packaging (Apache-2.0 or BSD-2-Clause), pyparsing (MIT), and python-dateutil
(BSD-3-Clause). Binary wheels may contain separately licensed runtime
components; their notices are provided by the upstream distributions.

This file is a practical summary, not a replacement for the authoritative
license text distributed with each third-party package.
