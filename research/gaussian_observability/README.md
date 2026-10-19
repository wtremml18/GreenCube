# Gaussian fluid reconstruction: observations, support and uncertainty

Read [RESULTS.md](RESULTS.md) for conclusions before any manuscript. This folder contains actual camera-based reconstruction experiments using public ScalarSyn images, together with a separate volumetric sensor-design audit. It extends the original thesis's sparse constrained-field question; it does not claim a new divergence-free kernel.

## Data and upstream implementation

- [LagrangianSplats paper](https://arxiv.org/abs/2605.09299), [official code](https://github.com/taoningxiao/LagrangianSplats), [released data](https://huggingface.co/datasets/tnx123/LagrangianSplats-Dataset).
- Pinned code: `180f61a8b56a39f3a71df168c54c87669081acff`.
- [Archive checksums](results/sources.json), [executed GPU environment and initial-checkpoint hash](results/gpu_environment.json).
- Images, volumes, upstream source and checkpoints live outside GreenCube. This repository stores new code, patches, sample definitions, predictions, metrics and figures. No private thesis/email content is included.

The upstream project has an MIT license, while several inherited renderer/submodule files carry separate research-use terms. Preserve those upstream notices and licenses when acquiring/building them. The patches here retain their source context; this experiment does not relicense upstream components.

## CPU experiment

Use the existing `../track_observability/requirements.txt` environment. Acquire the checksum-pinned archives with Python 3.12+ (or an earlier Python with tarfile's `data` filter):

```bash
python research/gaussian_observability/scripts/fetch_data.py \
  --upstream /path/to/LagrangianSplats --cache /path/to/download-cache
OPENBLAS_NUM_THREADS=2 python research/gaussian_observability/scripts/run_design.py \
  --data /path/to/LagrangianSplats/gt/scalarsyn --seeds 10
```

Repeat with `--tag short --length .08`, `--tag long --length .32`, `--tag discrepancy --sensor-sigma .3`, and `--tag independent --kind independent`. These five settings are reported together. `oracle_normal` explicitly uses ground-truth velocity to diagnose the density-to-velocity observation error; it is not a deployable image-only method. `density` uses released 3D density derivatives, not reconstructed camera density. `none` has no density observation.

```bash
python -m pytest research/gaussian_observability/tests -q
python research/gaussian_observability/scripts/summarize.py
```

## GPU experiment

Use a separate Python 3.10 environment with CUDA-compatible PyTorch, not the CPU package's Python 3.14 environment. The recorded execution used an RTX 5090, CUDA toolkit 12.8, PyTorch 2.9.1 with CUDA 12.8, and Taichi 1.7.4. [requirements-gpu.txt](requirements-gpu.txt) records the executed packages; install PyTorch/torchvision from the CUDA 12.8 wheel index. Install `setuptools<81` before building MMCV 1.7.2 with `--no-build-isolation`.

Clone the pinned upstream commit and initialize its submodules. From that checkout, apply `patches/upstream.patch`; from its `submodules/depth-diff-gaussian-rasterization` directory apply `patches/rasterizer.patch`. Build both CUDA extensions with `CUDA_HOME`, `TORCH_CUDA_ARCH_LIST` and `MAX_JOBS` appropriate to the GPU. The execution used `TORCH_CUDA_ARCH_LIST=12.0` and `MAX_JOBS=2`. On this WSL machine, `LD_LIBRARY_PATH=/usr/lib/wsl/lib` is required for Taichi to load the functioning Windows CUDA driver. Do not apply that path on unrelated systems.

The patches remove the obsolete scheduler `verbose` argument, add a missing `<cstdint>` header, repair an undefined variable in the window evaluator, and add an optional sparse velocity MSE term. A finite-difference check tests the new loss's upstream autodiff path. `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is used only with checkpoints generated locally by these runs.

From the upstream checkout, run its `main.py` with `--configs /absolute/path/to/configs/scalarsyn_images.py --skip_visualize --skip_render` to create the initial frame-zero model. The paired runs reuse its `window_0_4/gaussian_frame0` folder. In the executed work, the identical frame-zero model was first generated in a three-frame pilot, so its preserved directory is named `window_0_3/gaussian_frame0`.

Create sensor files with `scripts/image_sensors.py`, providing `--checkpoint` (the initial `chkpnt_coarse_3000.pth`), `--data` (GT directory), `--info` (camera `info.json`), and `--out`. The original control uses `--selection opacity`; the image-supported version uses `--selection silhouettes`; its support-matched random control adds `--placement random`. `--frames 3` produces readings for the three velocity intervals of a four-frame run. `--frames 9` prepares a ten-frame run.

Run `scripts/run_image_pairs.py` with `--upstream`, `--initial` (the model folder), `--sensors`, `--label`, `--weight`, and `--seeds 42 43 44`. The default is four frames; `--frames 10 --label visual16-f10 --seeds 42` is the longer-sequence sensitivity run. `--reuse-baselines` re-evaluates the existing image-only models from the `w0.01` runs instead of retraining them. The baseline uses zero sensor loss regardless of the weight in its filename. The full sequence of executed layouts and weights is documented in RESULTS.md.

After all runs:

```bash
cd research/gaussian_observability
python scripts/make_common_samples.py --data /path/to/LagrangianSplats/gt/scalarsyn
python scripts/sensor_only.py
python scripts/evaluate_common_motion.py --upstream /path/to/LagrangianSplats
python scripts/motion_components.py --data /path/to/LagrangianSplats/gt/scalarsyn
python scripts/image_roi.py --upstream /path/to/LagrangianSplats
python scripts/summarize.py
python scripts/figures.py --upstream /path/to/LagrangianSplats
python scripts/validate.py
```

These scripts establish common held-out motion samples across all layouts and foreground image scores. The full-frame upstream PSNR is retained separately. GPU arithmetic and sampling can vary slightly across runs/hardware; this is not a bitwise reproducibility claim.
