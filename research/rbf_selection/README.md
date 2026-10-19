# Selecting motion RBFs for dynamic Gaussian reconstruction

This experiment directly follows thesis Sections 2.4–2.6: center selection, orthogonal least squares (OLS), regularized OLS and shape/support parameters. It selects the **velocity-field basis**, not the appearance Gaussians or the sensors. The selected bases drive the pinned official LagrangianSplats implementation on the same public ScalarSyn inputs as the preceding study.

Read [METHOD.md](METHOD.md) for the equations and attribution, and [PROTOCOL.md](PROTOCOL.md) for frozen comparisons, declared exploratory follow-ups and limits. Read [RESULTS.md](RESULTS.md) for outcomes and the derivative-audit limitation. Neither greedy OLS nor the use of divergence-free kernels is claimed as new mathematics.

## What is implemented

- Block OLS selects an entire three-component vector kernel at a time, accounting for previously selected blocks. Unit tests compare every greedy choice against exhaustive least-squares refits and check rotation/scaling invariance and redundant columns.
- A dictionary of 192 image-supported centers at radii 30, 60 and 120 gives 576 candidate blocks. The corrected initial-geometry observation matrix composes renderer backward gradients with analytic RBF blocks and uses 8x8 pooled grayscale training images. Original central-difference pilots are retained separately after a failed derivative check. Three consecutive image differences supply interval targets. No released reference density or velocity volume enters selection.
- Image-only OLS and joint image/sensor OLS are compared against domain coverage, image-support coverage and random blocks at exact budgets 32, 64 and 128. Centers, radii and weights are all subsequently optimized; coefficients begin at zero in every case.
- A conventional sensor-location CV control chooses initial support scale. A separate regularized-OLS follow-up chooses a coefficient-prior strength by sensor-location CV before selecting blocks through the joint surrogate. All repeated times at each held-out sensor location stay in the same fold.
- Four upstream kernel parameter derivative checks pass, and the analytic kernel matches upstream GPU output. The renderer finite-difference audit fails and is retained. The corrected matrix passes four independent backward-operator assembly checks, establishing consistency with the optimizer without resolving the finite-difference discrepancy. None of these checks proves agreement with the entire nonlinear, grid-advected temporal training objective.

The initial rendering linearization holds appearance fixed, uses a direct one-step displacement at Gaussian centers, and reuses frame-zero geometry for all interval derivatives. It omits inflow and evolving appearance and approximates the upstream grid-based advection. Final reconstruction still uses the full upstream training code. The difference between surrogate fit and actual motion is an explicit research question.

## Reproduction

Prepare the pinned code, CUDA environment, archives and initial frame-zero geometry using [the previous module's instructions](../gaussian_observability/README.md). The existing patches in that module must already be applied. Apply this module's `patches/basis_initialization.patch` from the upstream checkout; it adds a basis-file override without changing the original path when that option is absent.

Use the GPU Python environment for the renderer and training scripts. On the recorded WSL host set `LD_LIBRARY_PATH=/usr/lib/wsl/lib`; use `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` only for locally generated trusted checkpoints. Scripts below accept absolute upstream and initial-model paths. Run from this module's directory:

```bash
# Historical finite-difference pilots and their shared controls (15 fits).
python scripts/build_dictionary.py --upstream /path/to/LagrangianSplats \
  --initial /path/to/gaussian_frame0 --derivative finite --cache /path/to/fd.npz
python scripts/run_suite.py --upstream /path/to/LagrangianSplats \
  --initial /path/to/gaussian_frame0 --methods poisson coverage random image_ols joint_ols

# Corrected backward matrix and selectors.
python scripts/build_dictionary.py --upstream /path/to/LagrangianSplats \
  --initial /path/to/gaussian_frame0 --derivative autodiff --cache /path/to/ad.npz
OPENBLAS_NUM_THREADS=2 python scripts/select_scale_cv.py
OPENBLAS_NUM_THREADS=2 python scripts/select_regularized.py
python scripts/run_suite.py --upstream /path/to/LagrangianSplats \
  --initial /path/to/gaussian_frame0 --methods image_adols joint_adols
python scripts/run_suite.py --upstream /path/to/LagrangianSplats \
  --initial /path/to/gaussian_frame0 --methods poisson coverage joint_adols --seeds 43 44
python scripts/run_suite.py --upstream /path/to/LagrangianSplats \
  --initial /path/to/gaussian_frame0 --methods poisson coverage image_adols \
  --budgets 64 128 --sensor-weight 0
python scripts/run_suite.py --upstream /path/to/LagrangianSplats \
  --initial /path/to/gaussian_frame0 --methods joint_rols --seeds 42 43 44

OPENBLAS_NUM_THREADS=2 python scripts/ablate_image_selection.py
OPENBLAS_NUM_THREADS=2 python scripts/sensor_only_control.py
python scripts/evaluate_common.py --upstream /path/to/LagrangianSplats
python scripts/audit_basis.py --upstream /path/to/LagrangianSplats
python scripts/summarize.py
OPENBLAS_NUM_THREADS=2 python scripts/diagnose_surrogate.py
python scripts/validate.py
python -m pytest tests -q
```

`build_dictionary.py` saves the finite-difference dictionary at `results/linearized_dictionary.npz` and the corrected matrix at `results/autodiff_dictionary.npz`; regenerating deliberately replaces the corresponding artifact. `select_regularized.py` reads the corrected matrix. The optional `build_dictionary.py --check-only` diagnostic records the finite-difference/kernel derivative audit and intentionally exits nonzero on the recorded renderer discrepancy. It is not a passing validation gate. The corrected build writes `autodiff_assembly_check.json` separately. An additional `--check-only --stability-sweep` run uses larger displacements and float64 probe accumulation, writes `renderer_stability_sweep.json`, and also fails its predeclared 5% criterion on the recorded case. Neither failed renderer diagnostic is presented as a passing test. The common motion targets and sensor values are inherited from `../gaussian_observability/results/`. The scripts never use held-out vector truth to select kernels. The suite records completed runs individually and skips them when resumed; use a fresh output location or archive previous outputs for an independent full rerun.

Raw volumes, upstream code and checkpoints are outside GreenCube. Saved dictionaries, selection targets, basis specifications, predictions, timing and failures/negative comparisons are retained. Selection/Jacobian cost is reported separately from optimization and total training-process time. Initial Gaussian training is shared across conditions and excluded from these incremental timing comparisons. Small control-preparation overhead and general process startup for dictionary construction are not fully profiled, so the cost plot is not a complete end-to-end speed benchmark.

This is a one-scene, four-frame exploratory experiment with simulated sensor readings and shared initial geometry. Optimization seeds do not establish independent-scene generalization. The broad-domain control uses a farthest subset of the upstream Poisson point set to enforce an exact budget, plus the upstream support-radius formula; it is labelled accordingly rather than presented as an unmodified published benchmark.

The completed results include 54 nonlinear camera fits, six of which are retained finite-difference pilots, plus ten direct linear sensor-regression fits. The regularized selector repeats all three budgets at three optimization seeds. `ablate_image_selection.py` removes image-loss rows while retaining image-derived candidate geometry; the same selected kernel sets do not count as new reconstruction fits. `sensor_only_control.py` reports full-bank, selected-subset, domain-coverage and three matched-size domain-bank controls. See the chronological protocol before interpreting follow-ups as primary comparisons.
