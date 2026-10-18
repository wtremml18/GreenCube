# Research conclusions before drafting a paper

**Date:** 2026-10-18. **Status:** executable research package and exploratory findings; no manuscript drafted or submitted.

The thesis contains a useful foundation, but the defensible next contribution is not a new divergence-free interpolation formula. The strongest direction emerging from these experiments is **determining what sparse, physically constrained reconstructions can identify, testing how much trustworthy geometry adds, and exposing ambiguity that a visually plausible field can hide**. This applies to both auroral reconstruction and dynamic Gaussian representations.

There is promising material here, but these results do not yet establish a submission-ready novel method or an independently validated auroral application. In particular, the 4D Gaussian connection is already active prior art: FreeGave (CVPR 2025) and LagrangianSplats (SIGGRAPH 2026) use divergence-free Gaussian motion. The latter directly combines divergence-free kernels and Gaussian advection. See the [literature audit](LITERATURE.md).

**Follow-up:** Public Lynch-group auroral inputs have now been located and inspected, including 195 ISINGLASS flow samples and Swarm/PFISR files. The [auroral data audit](auroral_data/README.md) adds close 2024–2025 application predecessors omitted from the initial search. This supplies a concrete route to application validation; it does not change the benchmark outcomes or establish an auroral accuracy result.

## What was recovered and built

The recovered thesis is Will Tremml's *Methods of Approximating Divergence-Free Vector Fields for Ionospheric Data*, Dartmouth, June 5, 2018. It was read from the original 77-page PDF, rather than inferred only from the supplied review. The [thesis assessment](track_observability/notes/thesis_review.md) distinguishes established contributions, mathematical corrections and missing validation.

The GreenCube repository contains acquisition, calibration and visualization code, but the original DivFree2/DivFree3 interpolation implementation was not recovered. All new code is isolated in `research/track_observability` and `research/gaussian_transport`. The private PDF and email metadata are excluded from the public repository.

The package includes independent 2D/3D vector kernels, properly specified regularization, nested validation, finite-basis ambiguity diagnostics, a Piola-transformed weighted-flux model, Gaussian deformation-gradient transport, exact density-ambiguity examples, three public datasets, raw results and figures. Nine principal CSVs contain **1,484 fitted-model evaluation rows**, with additional analytical and numerical audits. Eleven mathematical/numerical tests cover kernel properties, differential constraints, input invariances, ambiguity budgets, background recovery and transport.

## 1. The dimensionality criticism is consequential

For the analytic point dipole, the full 3D field is divergence-free, but the two-component meridional slice is not. Numerical derivatives agree with the exact slice divergence to about 4.4e−9. A model can therefore enforce its constraint perfectly while being structurally unable to represent the target.

Ten independent sampling seeds, each with 96 training points and 600 test points, gave:

| Domain and noise | Divergence-free MSE | Independent-component MSE | Interpretation |
|---|---:|---:|---|
| Wrong 2D slice constraint, clean | 0.02299 | 0.0000371 | About 620 times greater MSE under the wrong constraint |
| Wrong 2D slice constraint, 5% noise | 0.02146 | 0.000861 | About 25 times greater MSE |
| Full 3D field, clean | 0.000822 | 0.001014 | DF improves mean MSE by 19% |
| Full 3D field, 5% noise | 0.002370 | 0.003160 | DF improves mean MSE by 25% |

The common-length Helmholtz mixture performs better still in these tests. This does not prove the true field has a non-solenoidal component: the source-free dipole is also curl-free, and predictive model selection does not uniquely identify a physical Helmholtz decomposition. The initial single-seed test did not favor DF in 3D; the broader results above replace that impression rather than hiding it.

This is strong evidence to redo the thesis's dipole benchmark with the correct physical dimension. It is **not a diagnosis of the original MATLAB implementation**, which remains unavailable. Input-order and isotropic-unit invariance errors in the new implementation are below 5e−14; rotation equivariance is tested as well. The old coefficient-index smoother itself has not been reproduced or cleared.

Evidence: [baseline audit](track_observability/results/baseline_audits.json), [120 robustness outcomes](track_observability/results/dipole_robustness.csv), [summary](track_observability/results/dipole_summary.csv).

## 2. More samples along the same tracks can give false reassurance

An explicitly constructed smooth divergence-free field is zero along all four complete measurement tracks and nonzero between them. Its amplitude cannot be determined from those measurements, no matter how densely the tracks are sampled. The construction is an instance of standard nonidentifiability, not a new theorem.

Across 40 independent Fourier-field/noise seeds, increasing observations from 48 to 192 reduced selected random-fold validation MSE from **0.1033 to 0.00772**, about thirteenfold. Actual clean between-track MSE stayed near **0.106–0.108**. At 96 observations, DF still outperformed independent-component interpolation by **57.6%** (paired bootstrap 95% interval 48.5–66.7%). Thus a helpful physical prior and a severe observability limitation coexist.

Leave-track-out tuning did **not** solve the problem: it increased mean target MSE from 0.1070 to 0.1694 in this benchmark. Its prediction task differs from the target interpolation task. “Always use blocked CV” would be an unsupported conclusion.

The finite Fourier audit retains a 12-dimensional nullspace at cutoff four even as along-track density increases. Tiny coordinate jitter removes exact numerical nullity, but a noise-budget sensitivity calculation remains large. The diagnostic is explicitly conditional on a chosen basis and coefficient budget; it is not a confidence interval or an upper error guarantee.

Adding eight off-track points instead of eight additional on-track points reduces MSE by 91.5% in the deliberately structured synthetic setting. This large effect is **not universal**, as the public-fluid control below demonstrates.

Evidence: [primary summary](track_observability/results/synthetic_summary.csv), [paired intervals](track_observability/results/paired_summary.json), [noise audit](track_observability/results/noise_observability.csv), [validation figure](track_observability/figures/validation.png).

## 3. Arc geometry is useful when correct and damaging when wrong

The implemented extension respects the planar electrostatic relation **div(Bv)=0**, using an explicit curved coordinate map and a Piola-transformed covariance. It jointly fits a constant background in reference flux coordinates. The transformation and weighted constraint are established mathematics; the experiment tests their use here.

Twenty independent synthetic fields, each with 96 observations, gave:

| Geometry supplied to the model | Mean vector MSE |
|---|---:|
| Correct curvature and B | 0.05090 |
| Underestimated curvature | 0.05596 |
| Straight geometry, correct B | 0.09141 |
| Reversed curvature, correct B | 0.26513 |
| Straight, ordinary divergence-free velocity | 0.09745 |

Correct geometry reduces MSE by **44.3%** against straight geometry (paired 95% interval 31.3–56.2%). Reversed curvature produces **2.90 times** the error. These effects support a focused geometry-reliability study.

However, this is an oracle-geometry benchmark: fields were generated in known curved coordinates, and B is prescribed. No image segmentation, auroral event, conductance model, inductive correction or independent crossing was used. We have not demonstrated an auroral improvement, calibrated uncertainty, or superiority to Local Divergence-Free Fitting, SECS or Lompe.

Evidence: [geometry outcomes](track_observability/results/geometry.csv), [physical model](track_observability/src/geometry.py), [combined figure](gaussian_transport/figures/fluid_and_geometry.png).

## 4. Public data limit the broad claims

### Ocean observations: GLAD and LASER

Four checksum-pinned CSVs from the official Helmholtz-GP repository supply 1,183 GLAD and 240 LASER observations. The experiment uses subsets of 160, outer spatial-quadrant holdout, and inner spatial tuning. A follow-up broadens the length/regularization grid and repeats five subset selections.

| Dataset | DF error / training-mean baseline | Independent | Common-length Helmholtz |
|---|---:|---:|---:|
| GLAD, average over repeated subsets | 1.049 | 1.051 | 1.049 |
| LASER, average over repeated subsets | 0.821 | 0.691 | 0.555 |

Lower is better; one means equal to the simple training-mean predictor. GLAD shows no aggregate improvement. LASER favors allowing a mixed field, with substantial subset sensitivity. The initial narrower grid gave different magnitudes, so a general “DF wins on real data” claim is untenable.

These released CSVs omit buoy identity and timestamps and use temporally collapsed preprocessing. The test is exploratory location holdout, not independent drifter/event validation. The comparator is not a full reproduction of the published Helmholtz-GP model.

Evidence: [provenance](track_observability/data/README.md), [robustness summary](track_observability/results/public_robustness_summary.csv).

### Public 3D smoke: ScalarSyn

The 704 MB LagrangianSplats reference archive was downloaded and its upstream SHA-256 verified. Four selected frames provide full 3D density and velocity fields. At each frame and sampling seed, models receive 96 synthetic direct velocity measurements and are tested at 512 disjoint smoke-volume voxels. This tests interpolation on realistic simulated flow structure, **not reconstruction from video**.

The initial lines missed most visible smoke. Their apparent 56% benefit from replacing eight observations was confounded by support coverage. After requiring **all training observations in every design to lie inside visible smoke**, and using common noise amplitude across designs:

| DF sampling design, fixed 96 observations | Mean vector MSE | Mean MSE / target velocity energy |
|---|---:|---:|
| Four lines inside smoke | 0.6950 | 0.4612 |
| Replace eight line samples with volume samples | 0.6554 | 0.4358 |
| Sample all 96 throughout smoke volume | 0.3563 | 0.2326 |

Replacing eight yields **5.7%** improvement, with frame-wise effects of 2.6–8.4%. Fully volumetric sampling reduces mean MSE by about **48.7%**. The DF advantage over independent kernels is also smaller for volumetric sampling than for line sampling.

This control materially weakens the dramatic synthetic effect while preserving the broader lesson about coverage. The four frames are from one sequence; five noise/sampling repeats per frame do not create twenty independent fluid events. Stored velocities have nonzero finite-difference divergence. No formal population confidence interval is attached.

Evidence: [protocol including the confound correction](gaussian_transport/PROTOCOL.md), [support-matched outcomes](gaussian_transport/results/fluid_support_matched.csv), [frame summaries](gaussian_transport/results/fluid_frame_summary.csv).

## 5. The 4D Gaussian connection is real, but the proposed basic combination is already published

For time-varying 3D Gaussians, a velocity model can transport centers and local covariances using dmu/dt=v and dF/dt=(grad v)F, with Sigma=F Sigma0 Fᵀ. Incompressibility preserves det F in continuous time. This can provide physically meaningful local volume behavior for suitable fluids. It is not a universal constraint on arbitrary dynamic scenes, nor does it automatically conserve renderer opacity or appearance.

Twenty analytic field/noise seeds gave:

| Motion | DF center RMSE | Independent center RMSE | Conclusion |
|---|---:|---:|---|
| Truly solenoidal | 0.00369 | 0.00400 | Small 7.7% average benefit; 95% interval −1.5% to 16.7%, inconclusive |
| Same flow plus isotropic expansion | 0.08729 | 0.00421 | Wrong incompressibility assumption causes about 20.8 times the error |

RK4 convergence tests distinguish integration error from the continuous constraint. Covariance transport is exact for affine motion and only locally approximate for nonlinear deformation. These results do not measure rendered PSNR, novel-view quality or performance against published Gaussian methods.

An exact counterexample is more revealing: an anisotropic Gaussian density admits non-rigid, divergence-free affine flows that leave its full density unchanged for all time while moving material points. At t=1, marker RMS displacement is 0.7675 in the chosen units, while covariance changes by only 5.6e−17. Density-only images with fixed appearance cannot distinguish these flows even with perfect views. On public density volumes, tangent perturbations also change the instantaneous advective derivative by only about 1.4e−14 at unit vector RMS.

This is a concrete manifestation of the known aperture/identifiability problem. It does not show that all such alternatives obey the same momentum equations, boundaries or textured appearance. Those can supply additional constraints. It does show why rendering fidelity and low divergence alone cannot certify material motion.

Evidence: [Gaussian outcomes](gaussian_transport/results/transport.csv), [exact ambiguity and integration checks](gaussian_transport/results/exact_ambiguity.json), [derivation](METHODS.md), [density-ambiguity figure](gaussian_transport/figures/density_ambiguity.png).

## Publication decision

**Do not submit the thesis or this package as evidence of a new divergence-free kernel, a first divergence-free Gaussian reconstruction method, or a validated auroral reconstruction improvement.** Those claims are unsupported or contradicted by prior art.

The most promising next question is: **Can a reconstruction system identify its weakly observed motion and use uncertain geometry or a small number of strategically placed independent measurements to reduce the consequential ambiguity?** The current package gives reproducible counterexamples, useful diagnostics and conditional geometry improvements, but not a verified novel algorithm or end-to-end solution.

For an auroral paper, the decisive missing evidence is multiple independent tracks/events with independently observed morphology, realistic magnetic geometry, and a relevant ionospheric baseline. For a Gaussian paper, it is an actual image-based reconstruction experiment with an established renderer, temporal/novel-view evaluation, velocity/deformation truth, geometry/appearance uncertainty, and a comparison against FreeGave or LagrangianSplats. A claim about uncertainty needs measured coverage or defensible bounds under explicit assumptions; the present sensitivity radius is neither.

The strongest result of this investigation is a more precise research target and a tested foundation for pursuing it. The manuscript should follow that evidence, rather than turn exploratory successes into a novelty claim prematurely.
