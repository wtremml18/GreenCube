# Auroral fields linked to public optical observations

This is an **executed exploratory application** of the constrained-kernel research to Kristina Lynch's public research data. It links measured fields to auroral photography, fits physical and image-informed models, and evaluates held-out measurements. It is not a manuscript or a validated two-dimensional auroral field product.

Start with [RESULTS.md](RESULTS.md), [actual camera photographs](figures/actual_auroral_photographs.png), [all six optical geometries](figures/six_event_geometry.png), and the [ISINGLASS reconstruction](figures/isinglass_maps.png). Numerical predictions, split membership, hyperparameters, quality selections and conditional field maps are in `results/`.

## Public data and attribution

The [Dartmouth Lynch archive](https://rcweb.dartmouth.edu/LynchK/) supplies the inputs; [sources.json](sources.json) pins all 17 files (576,639,179 bytes). Raw files remain outside Git in `../auroral_data/data/`. No email or private thesis content is included. Retrieval date: 2026-10-18. Hashes identify downloaded bytes, not a publisher-signed release. Please cite the original investigators and dataset providers when reusing outputs.

- **ISINGLASS, 2 March 2017:** 195 processed flow vectors from `flow_data_ig.mat`; 130 lie inside the optical inversion grid. `particles_ig.mat` contains 136 time-indexed energy-flux maps on a 256 × 256 geomagnetic grid. These are optical inversion products, not direct field measurements or 136 independent events. Main map: 07:54:10 UTC. No track replication is used.
- **Swarm-over-Poker campaign, six conjunctions in February–March 2023:** use the public PFISR electric-field estimates and DASC calibrated 428 nm images. The executed field benchmark uses radar, not Swarm drift measurements: the earlier audit did not establish suitable component-wise satellite quality and coordinate mapping. Six 512 × 640 optical products, including their published geographic maps, are loaded. PFISR estimates are mapped once to the matching 110 km altitude, not counted again at their other mapped altitudes.
- **Photography:** three original archived PNG exposures from 10 February 2023 at 09:51:25, :28 and :30 UTC (428, 558 and 630 nm) are displayed separately. They are monochrome filter exposures, not an RGB photograph. The geometry model uses the provider's calibrated, denoised/coadded blue image and geographic registration, not a manually traced curve or a generated picture.

References:

1. Clayton, Lynch et al. (2019), [Two-Dimensional Maps of In Situ Ionospheric Plasma Flow Data Near Auroral Arcs Using Auroral Imagery](https://doi.org/10.1029/2018JA026440).
2. Clayton, Burleigh, Lynch et al. (2021), [Examining the Auroral Ionosphere in Three Dimensions Using Reconstructed 2D Maps of Auroral Data to Drive the 3D GEMINI Model](https://doi.org/10.1029/2021JA029749).
3. Burleigh, Lynch, Zettergren & Clayton (2022), [Spatiotemporal Limitations of Data-Driven Modeling: An ISINGLASS Case Study](https://doi.org/10.1029/2021JA030242).
4. van Irsel, Lynch, Mule & Zettergren (2024), [Generation of Top-Boundary Conditions for 3D Ionospheric Models Constrained by Auroral Imagery and Plasma Flow Data](https://doi.org/10.1029/2024JA032722).
5. van Irsel, Lynch et al. (2025), [Current Closure and Joule Heating in Data-Driven 3-D Auroral Arc Simulations](https://doi.org/10.1029/2025JA034311).

The 2024 method is a required future published comparator. Our image-potential model is an independent implementation of established constructions, not a numerical reproduction of that entire pipeline. Neither kernel differentiation nor the combination of auroral geometry and electrostatics is claimed as new.

## Model and coordinates

Fit an electric potential Gaussian process for radar fields. In image coordinates `q=(x,y-f(x))`, `f` is a quadratic optical brightness ridge. The covariance of potential derivatives is pushed to the physical chart by `Dq^T`, so the resulting electric field is curl-free in that chart. For rocket velocity, rotate that gradient by 90 degrees, obtaining a planar divergence-free field under the constant-background-field approximation. A constant field is jointly fitted by generalized least squares; there is no ordered-coefficient penalty or empirical 120% background subtraction.

For radar, the chart is `x=R cos(lat0) (lon-lon0)`, `y=R (lat-lat0)` with `R=6481 km`, `lat0=65.12 degrees`, `lon0=-147.43 degrees`. A potential's chart derivatives are covector components: `E_chart,x=cos(lat)/cos(lat0) E_east`, `E_chart,y=E_north`. Both measurements and covariances are transformed before fitting; predictions and losses return to physical east/north units. This corrects the latitude-dependent longitude metric. A spherical Earth and constant-altitude surface remain approximations.

The rocket uses the provided geomagnetic position/vector columns in a local nominal 100 km plane. The decoded upstream note explicitly describes an approximate rotation, not a proper Apex transformation. This limits physical interpretation of small derivative errors. No magnetic field magnitude is invented and no electric-field unit conversion is claimed for the rocket flow results.

The image ridge uses only optical pixel positions and brightness: 25 east-west bins; squared brightness excess above each bin's 80th percentile; quadratic regression with four robust reweighting iterations. Radar image ROI is x=[-180,180] km, y=[-100,420] km. The rocket uses its full optical grid. This is a deliberately simple morphology proxy. Multiple arcs, horizon artifacts and truncated arcs can defeat it; the six-image figure exposes these cases.

Seven controls share the same search grid where applicable:

| Model | Spatial assumption |
|---|---|
| Constant | Training-only arithmetic mean |
| Unconstrained | Independent isotropic Gaussian components |
| Physical / isotropic | Potential-derived electric field; rotated-gradient rocket velocity |
| Physical / straight | Same physical kernel, 4:1 east/north correlation-length ratio |
| Image / linear | Image ridge slope, curvature removed; 4:1 ratio |
| Image / curved | Full image-only quadratic ridge; 4:1 ratio |
| Image / reversed | Negated ridge geometry; negative control |

The full observation covariance is used for radar. Gaussian amplitudes and added noise floors are tuned on training data. Rocket measurement errors are absent from the source; 100 m/s is an explicit provisional assumption, with 50 and 200 m/s sensitivity checks.

## Evaluation and interpretation

`design.json` records the initial exploratory design. It was written before first model losses, but is **not a preregistration**. Subsequent coordinate auditing corrected the spherical electric-field metric; a timestamp audit added the explicitly marked temporal sensitivity analysis. Final results include these corrections, and no outcome-based event selection is used.

Select the radar integration containing the optical timestamp, and retain bins with finite, positive horizontal covariance, both marginal standard errors <30 mV/m, at least 30 contributing inputs, and reduced chi-square between 0 and 3. Selection uses quality metadata, not measured field magnitude. Sensitivity uses 20 and 50 mV/m error thresholds.

Four contiguous outer holdouts exclude an additional neighboring original bin (two original rows for the rocket). Three contiguous inner groups tune correlation length, amplitude and noise floor using only outer-training data. Every outer test prediction is saved. Geometry uses independent optical data, not held-out field values. Primary loss is horizontal vector RMSE in mV/m or m/s; the radar events are summarized equally. Direction error and 95% ellipse coverage are also saved.

The covariance is conditional universal-kriging uncertainty, including estimation of the constant background. Held-out coverage adds reported measurement error and fitted noise floor. It **does not** include uncertain geometry, evolving aurora, unknown instrument systematics or hyperparameter uncertainty. The maps export conditional *latent* covariance separately from measurement noise. The ellipse threshold is chi-square(2), 0.95 = 5.991464547.

Two essential boundaries on claims:

- Held-out radar bins are already processed, spatially aggregated inversion products. Cross-bin error correlations and full line-of-sight observation operators are not supplied here. This is interpolation of processed products, not a raw radar inversion or independent-instrument validation.
- A radar estimate averages roughly five minutes; its paired optical snapshot is much shorter. The static rocket reconstruction similarly spans time. The pinned MATLAB datetime payload is decoded and verified against the documented launch plus elapsed seconds to within 1 ms. We additionally evaluate +/-30 s rocket windows at optical frames 0,45,90,135. These overlap and must not be counted as four independent events.

Maps away from a track remain model-dependent extrapolations. In map figures, predictions more than 50 km from measurements are faded. The accompanying uncertainty and model-disagreement figures make this lack of support visible. They do not make the uncertainty calibrated.

## Reproduce

Use Python 3.11+ and the packages in `requirements.txt`. Commands from the repository root (set `PYTHON` to your environment's Python):

```bash
python -m pip install -r research/auroral_reconstruction/requirements.txt
python research/auroral_reconstruction/fetch.py
python -m unittest discover -s research/auroral_reconstruction -p 'test_*.py' -v
python research/auroral_reconstruction/run.py --sensitivity
python research/auroral_reconstruction/summarize.py
```

The experiment runs on CPU with BLAS limited to one thread. `fetch.py` fails on changed source hashes. Tests check the physical derivative constraints, covariance positivity, permutation invariance, background recovery, guarded nested splits, optical ridge extraction, geographic metric and the decoded timestamp anchor. No external MATLAB routines are executed. No investigators have been contacted and no paper has been drafted.
