# Review of Will Tremml's Dartmouth thesis

**Source:** *Methods of Approximating Divergence-Free Vector Fields for Ionospheric Data*, dated June 5, 2018, 77 PDF pages (69 numbered body/bibliography pages). Examining committee: Kristina Lynch (chair), Anne Gelb, Robyn Millan. The PDF was recovered from the author's Gmail. The original PDF and email metadata remain outside the public repository.

## What the thesis establishes

The strongest contribution is an accessible, working synthesis of matrix-valued divergence-free radial basis functions, shape selection, regularization, center selection, and magnetometer-array acquisition. Equations 2.18–2.20 give the familiar solenoidal matrix kernel; the two- and three-dimensional tools are described as DivFree2 and DivFree3. The thesis appropriately acknowledges that it does not claim new mathematical theory (printed pp. 58–59).

The controlled examples distinguish clean and noisy target fields and report cases where natural-neighbor interpolation wins. This is useful evidence, although some broad statements in the abstract/conclusion are stronger than the detailed noisy comparisons. The hardware work provides a convincing qualitative demonstration, and the thesis explicitly recognizes that calibrated quantitative validation remains necessary.

## Corrections needed for a research paper

1. **The electrostatic inference needs extra assumptions (printed p. 1).** The expansion of curl(-v×B) omits -(B·∇)v+(v·∇)B. Electrostatic E and divergence-free B alone do not imply divergence-free v. A locally uniform B and negligible variation of v along B are sufficient additional conditions for the displayed reduction. A magnetic-field constraint is exact in three dimensions; an assumed planar velocity constraint is conditional.
2. **Two-dimensional slices need care.** A 3-D divergence-free magnetic field obeys ∂xBx+∂yBy=-∂zBz on a slice. A planar component field need not be divergence free. This is an additional possible explanation to investigate for the constructed dipole discrepancy; it is not a retrospective diagnosis without the actual original code and coordinate convention.
3. **Divergence/curl ratios do not validate accuracy (pp. 40, 54).** A hard-constrained interpolator will satisfy the imposed differential relation even when it predicts the wrong field. Constraint satisfaction must be reported separately from independent value/gradient errors.
4. **Replicated data are not independent measurements (pp. 40, 53).** The auroral example horizontally repeats a single vertical crossing along an inferred arc. Cross-validation within this constructed dataset evaluates the replication/interpolation assumptions, not independent knowledge of the intervening ionosphere. Interpolated pseudo-labels should not be counted as additional observational evidence for downstream ML.
5. **The confidence calculation does not establish MSE precision (p. 25).** A binomial interval for inclusion frequency cannot provide a 95% confidence guarantee on continuous squared prediction error. Repeated folds also share observations. Quantify uncertainty over independently generated fields, experiments, or appropriate spatial/temporal clusters.
6. **Background correction is case-specific (pp. 26–29).** The suggested 120% correction comes from a particular constructed example, and is not a generally justified estimator. Fit a mean or physically motivated background using training data only; propagate uncertainty where it matters.
7. **Regularization equations require an explicit objective (pp. 29–31).** (K+λI)α=y is a kernel-ridge normal equation for an RKHS penalty when K has the relevant properties. Ordinary coefficient-penalized least squares instead yields (AᵀA+λDᵀD)α=Aᵀy. Do not interchange these objectives, or impose coefficient-neighbor differences without defining a meaningful ordering.
8. **Narrow technical statements.** Wendland C2 is twice continuously differentiable, not non-continuously differentiable (p. 11). Kernel admissibility is not guaranteed by arbitrary continuity. A dense n×n factorization generally costs O(n³), and explicit inversion is unnecessary. Convex-hull membership alone does not certify reconstruction accuracy. The scalar RBF universal-approximation citation does not by itself establish all properties of a constrained vector kernel.

## Repository and reproducibility assessment

GreenCube at base commit 18e2d62 contains Arduino acquisition, MATLAB calibration/visualization, and practice magnetometer files. Neither the checked-out Master tree nor the available filename inventory contains DivFree2 or DivFree3. The 2018 'Thesis and Materials' email references those tools but has HTML attachment placeholders, not downloadable MIME attachments. The new Python implementation is therefore an independent implementation of the published kernel construction, **not a tested port or reproduction of the original MATLAB results**.

The practice magnetometer files do not provide sufficient verified experiment labels, calibrated ground truth, geometry, and current metadata for a new quantitative magnetic-field claim. They were inspected for context, not relabeled as validation data.

## Research direction selected

Investigate transverse non-identifiability under parallel-track sampling: distinguish a valid differential constraint, prediction at observed-track locations, and recovery inside unsampled gaps. Provide a constructive counterexample, a finite-basis ambiguity audit, a deterministic noise-budget extension, synthetic evaluation, and a public-data stress test. The work is methodological and deliberately does not claim that the old auroral dataset or the present experiments establish a new ionospheric discovery.
