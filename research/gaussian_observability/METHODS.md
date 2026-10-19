# Experimental specification and limits

These are exploratory experiments. The sensor-loss weight and support rule were revised after pilot results, and all completed controlled variants are retained. The reserved camera is excluded from training and sensor placement, but these development results are not an untouched final model-selection test. No novel kernel, visual-hull construction, or Gaussian conditioning formula is claimed.

## Public data and coordinate audit

The released ScalarSyn arrays have shape `(100,178,100,channels)` with axes `(z,y,x)` and vector components `(x,y,z)`. We follow the upstream configured crop `(100,150,100)`. The renderer's simulation coordinates have extents `(100,150,100)` even when its computational grid is reduced by `scale=2`. Its effective frame time is `sim_steps/scale=2/2=1` in native simulation units. An initial density-derivative pilot incorrectly used 2; those provisional files were discarded and all five reported CPU settings were regenerated with 1.

The camera transform convention follows the pinned loader's executed code, not its misleading COLMAP comment: the stored matrix is treated as camera-to-world with OpenGL axes, then y and z camera axes are flipped before inversion. `src/visibility.py` reproduces this convention. Unit tests check the optical axis, vertical sign, camera translation and the anisotropic voxel/world round trip.

GT is used for synthetic instrument readings and evaluation. Its released density is additionally used for the **separate volumetric CPU diagnostic**, explicitly an oracle-volume experiment. GT density and velocity do not determine camera-experiment sensor placement.

## Camera reconstruction and synthetic instruments

The initial Gaussian model is fitted to frame zero from four training cameras for 3,000 iterations. One camera is reserved. All controlled GPU runs share this initial image-fitted model; seeds 42, 43 and 44 vary subsequent optimization/sampling, not independent scenes or independent initial geometry fits.

Each four-frame run uses one window, 200 velocity/appearance optimization epochs, the configured 20,000-kernel budget before `scale^3` reduction and Poisson sampling (about 1,121 actual centers), Wendland C4 divergence-free kernels, velocity L1 weight 0.01, and the upstream vorticity loss weight 0.1. The source's vorticity-frame selector becomes active at four frames. The ten-frame sensitivity run uses the same budget and weights with nine reconstructed velocity intervals, one optimization seed. These budgets are below the published full benchmark settings.

The instrument loss is the mean squared vector-component error at fixed locations, added to the upstream direct velocity losses. The tested weights are 0.01 and 1 for opacity-only placement, and 1 for the image-supported layouts. Noisy readings equal trilinearly interpolated released velocity plus independent Gaussian component noise with sigma 0.05. A four-frame run has **16 fixed locations × 3 observation times = 48 vector readings**; the ten-frame run has 144. This is a simulated measurement-budget experiment, not real laboratory instrumentation.

Placement variants:

1. **Opacity-only control:** keep the upper half of the initial Gaussians by `opacity × product(scales)`, within an interior box, then use farthest-point coverage. The first point is nearest the candidate mean. This selected almost entirely empty space. Multiplying by the fitted DC emitted brightness and retaining 90% of that proxy's total mass produced the same selected points in this pilot; it did not fix visibility.
2. **Image-supported coverage:** retain Gaussian centers whose projections have intensity above 1% of the corresponding image maximum in **all four training images at frame zero**, within the same interior box. This is a visual-hull-style support filter. Apply the same farthest-point rule to the 2,892 retained candidates.
3. **Support-matched random:** draw 16 without replacement from that exact candidate pool. Five layout seeds, 413–417, are retained. Layout 413 is fitted with three optimization seeds; the additional layouts use optimization seed 42. These are distinct from the optimization seeds.

The sensor loss's gradient through upstream Taichi/CUDA kernels is checked against central finite differences for five high-magnitude weight derivatives. Maximum relative error is recorded in `results/gpu_gradient_check.json`.

## Motion and appearance evaluation

`make_common_samples.py` selects 4,096 GT smoke voxels per velocity frame, with density above 1% of the frame maximum, away from the inflow and domain edges. Every target is more than two native voxels from **every sensor location in every tested layout**. All methods share those targets and GT vectors. Locations are frozen with seed `7400+frame`. `common_motion.csv` and `common_motion_predictions.npz` supersede the earlier layout-specific motion samples retained in each original camera-run JSON.

Vector RMSE is `sqrt(mean_voxel(sum_component(error**2)))` in native simulation velocity units. Tables report the mean of these per-frame RMSEs, and, when applicable, optimization seeds. Repeated baseline evaluations and overlapping frames are not treated as independent experiments.

The upstream window evaluator produces PSNR, SSIM and LPIPS on the reserved camera for every evaluated frame. A recorded patch repairs its undefined frame-count variable; evaluation retains the true 30-frame timestamp normalization. A separate smoke-region evaluation computes PSNR and relative RMSE from the saved 8-bit images, using the bounding rectangle of GT pixels brighter than 1% of the frame maximum plus an eight-pixel margin. The ROI is an evaluation definition only. Its scores are not numerically interchangeable with the upstream floating-point full-frame scores.

A conventional sensor-only comparator fits the existing independent Python Gaussian RBF model to the same 16 values per time. Both divergence-free and unconstrained vector kernels are tested. Four-fold point CV selects length from `{.08,.16,.32,.64}` and ridge from `{.001,.03,.3}`, using no held-out velocities or temporal images. Coordinates are divided by 100 isotropically. Its sensor **locations** can still be image-informed; “sensor-only” refers to estimating velocity from values, not to acquiring geometry without cameras.

The optional post-fit directional diagnostic uses central finite differences of released density at the common withheld points. It excludes the lowest 30% of gradient magnitudes in each frame and magnitudes at most `1e-8`. With unit gradient normal `n` and vector error `e`, the normal squared error is `(n·e)^2` and the tangent squared error is `||e||²-(n·e)^2`. The normal has one dimension and the tangent plane has two; the CSV also records per-tangent-dimension error. This GT-derived diagnostic neither changes predictions nor proves a nullspace of the complete temporal rendering problem.

## Volumetric observation-design diagnostic

Frames 1, 10, 20 and 28 are evaluated with ten sampling seeds. Each trial has 256 scalar density-derived observations, 192 possible vector-sensor locations and 256 held-out vector targets, all distinct voxels inside density support. Gradients use centered spatial differences; scalar observations use the centered temporal derivative divided by gradient magnitude. The lowest 30% of support gradient magnitudes are excluded from scalar observations.

For normal vector `n=grad(rho)/|grad(rho)|`, the measurement model is `n·v = -(rho[t+1]-rho[t-1])/(2|grad(rho)|)`. It assumes scalar advection with the effective unit frame time. The `oracle_normal` diagnostic substitutes the exact normal component from GT velocity; `none` omits density observations entirely.

The velocity covariance is `4 K_df + 4 I`, with the last term a shared random constant background (not independent observation noise). The primary length is .16 after coordinates are divided by 100. Scalar observation sigma is .3. Simulated vector noise is .05. The discrepancy sensitivity case assumes vector sigma .3 despite actual .05; this tests extra model-error allowance, not a noisier instrument.

After scalar conditioning, the integrated-variance rule selects the candidate block maximizing

`trace(C_target,sensor (C_sensor,sensor + sigma² I)^-1 C_sensor,target)`.

This is standard greedy A-optimal design for a Gaussian model. Selection uses target **coordinates** as its integration set, never target velocity labels. It is therefore a transductive placement evaluation. Competing designs are random, farthest-point coverage and largest pointwise posterior trace. The latter two also use no target labels.

All settings are retained: primary, lengths .08 and .32, enlarged assumed vector noise, and an unconstrained kernel. There are 12,000 metric rows, but only four correlated snapshots of one simulation, with repeated sampling—not 12,000 independent events. Bootstrap intervals average the four fixed frames within each seed and quantify only Monte Carlo sampling variability conditional on this scene.

## What these experiments cannot establish

The studies do not establish superiority over a full-budget LagrangianSplats run, a new interpolation method, a universal optimal sensor design, calibrated real-world uncertainty, or generalization to other scenes. Sparse-reading improvement may simply reflect added physical information; the sensor-only comparator tests this directly. The fluid assumptions do not transfer automatically to arbitrary deforming objects in general 4D Gaussian splatting, or to auroral brightness treated as a passive scalar.
