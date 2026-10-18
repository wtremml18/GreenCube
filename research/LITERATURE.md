# Literature and novelty audit

Checked against primary publications or official project sources on 2026-10-18. This is a targeted search, not a systematic review or proof of first-in-literature novelty.

| Source | Relevance and boundary on claims |
|---|---|
| [Vennell & Beatson, 2009, *A divergence-free spatial interpolator for large sparse velocity data sets*](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2008JC004973) | Sparse velocity reconstruction with solenoidal kernels predates the thesis. |
| [McNally, 2011, *Divergence-free interpolation of vector fields from point values*](https://arxiv.org/abs/1102.4852) | Direct magnetic-field RBF predecessor, including background-field considerations. |
| [Jidling, Wahlström, Wills & Schön, 2017, *Linearly constrained Gaussian processes*](https://papers.nips.cc/paper_files/paper/2017/hash/71ad16ad2c4d81f348082ff6c4b20768-Abstract.html) | Constraints can be built into covariance models; combining physics and learning is established. Our package does not yet validate posterior uncertainty. |
| [Roberts et al., 2017, *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*](https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881) | Splits must reflect the prediction task; blocking can change interpolation into extrapolation. Our leave-track result must not be sold as an automatic remedy. |
| [Bristow, Hampton & Otto, 2016, *High-spatial-resolution velocity measurements derived using Local Divergence-Free Fitting of SuperDARN observations*](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015JA021862) | A close ionospheric application comparator. We have not reproduced it or validated against SuperDARN observations. |
| [Berlinghieri et al., 2023, *Gaussian Processes at the Helm(holtz): A More Fluid Model for Ocean Currents*](https://proceedings.mlr.press/v202/berlinghieri23a.html) | Public GLAD/LASER data source and a relevant mixed-field model. Our common-length coarse mixture is a simpler comparator, not a reproduction of their full GP. |
| [Broeren, Klein & TenBarge, 2024, *Multi-spacecraft Magnetic Field Reconstructions: A Cross-Scale Comparison of Methods*](https://arxiv.org/abs/2310.15187) | Relevant reconstruction literature for anisotropic spacecraft sampling and numerical turbulence benchmarks. |
| [Ni et al., 2025, *Representing Flow Fields with Divergence-Free Kernels for Reconstruction*](https://xjr01.github.io/files/sca25dfk.pdf) | Contemporary use of established kernels, including Wendland C4. Neither divergence-free kernels nor their use for modern flow reconstruction is new here. |
| [Wu et al., CVPR 2024, *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering*](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.html) | Dynamic Gaussian rendering is the broader application context. Our transport benchmark does not implement its renderer or compare visual fidelity. |
| [Li, Song, Zhou & Yang, CVPR 2025, *FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity*](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_FreeGave_3D_Physics_Learning_from_Dynamic_Videos_by_Gaussian_Velocity_CVPR_2025_paper.pdf) | Already introduces divergence-free per-Gaussian velocity modeling for dynamic videos. This excludes a broad novelty claim about incompressible Gaussian motion. |
| [Tao, Chen & Chu, SIGGRAPH 2026, *LagrangianSplats: Divergence-Free Transport of Gaussian Primitives for Fluid Reconstruction*](https://arxiv.org/abs/2605.09299) | Direct prior art: continuous divergence-free kernels drive Gaussian advection, trained from sparse video observations using a sliding window. We use released ScalarSyn ground truth for a separate direct-velocity sampling study, not as a reproduction or head-to-head rendering comparison. [Official implementation](https://github.com/taoningxiao/LagrangianSplats), [public data](https://huggingface.co/datasets/tnx123/LagrangianSplats-Dataset). |
| [Horn & Schunck, 1981, *Determining Optical Flow*](https://www.sciencedirect.com/science/article/pii/0004370281900242) | The underdetermination of motion from scalar image variation is longstanding. Our Gaussian-density construction is a worked diagnostic, not discovery of the aperture problem. |
| [DefElement: mapping finite elements](https://defelement.org/finite-elements.html) | Documents the established contravariant Piola map J/det(J). The curved-coordinate construction uses this existing principle. |

The potential contribution is a carefully validated **application-specific observability and geometry-sensitivity study**. The nullspace/SVD calculation, generalized eigenvalue sensitivity calculation, Piola transform, covariance transport, and scalar aperture ambiguity are established mathematical tools. Packaging them does not by itself establish a publishable novel algorithm.

A convincing next paper would require a clearly scoped new claim and stronger application evidence: independent auroral tracks/events with independently measured morphology, or image-based fluid reconstruction with controlled geometry/appearance uncertainty and known velocity truth. The present work has not reached that point.

## Follow-up: Lynch-group application predecessors and verified data

The targeted search above omitted two particularly close application sources, now added after locating and opening their public data:

- [van Irsel, Lynch, Mule & Zettergren (2024)](https://doi.org/10.1029/2024JA032722), *Generation of Top-Boundary Conditions for 3D Ionospheric Models Constrained by Auroral Imagery and Plasma Flow Data*. A required geometry/electrostatic reconstruction comparator.
- [van Irsel et al. (2025)](https://doi.org/10.1029/2025JA034311), *Current Closure and Joule Heating in Data-Driven 3-D Auroral Arc Simulations*. Public Swarm-over-Poker inputs and simulations coauthored by Lynch.

See the [data audit](auroral_data/README.md) for directly verified files and limitations. These references further restrict novelty claims. Public data are available; the missing step is a validated application comparison, not merely finding an archive.
