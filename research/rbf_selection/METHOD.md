# Motion-basis selection through observations

The thesis's direct connection is its center/shape selection and OLS discussion (Sections 2.4–2.6), not a claim to have invented divergence-free kernels. This implementation adapts established block greedy regression to the observation map of an existing dynamic Gaussian reconstruction system. It changes initialization of the motion basis; the downstream method already learns centers, radii and vector coefficients.

## Representation and local objective

For interval t, write v_t(x) = sum_j K_j(x) w_jt, where K_j is a 3 by 3 divergence-free Wendland C4 matrix kernel, with candidate center c_j and support radius h_j. A selected block contributes three coefficients per interval. Once selected, each kernel has seven trainable parameters per interval: three center coordinates, one radius and three weights. The selected initial centers/supports are shared across the three intervals, but the subsequent optimization has independent interval parameters.

Let P be the frame-zero Gaussian positions, R_c the renderer for training camera c, and Q the grayscale/pooling operator. The local image dictionary is

    A_cj = Q D_P R_c(P) K_j(P).

Coordinate transforms from native voxel positions to world coordinates are included in the derivative. The targets are Q(I_{t+1,c} - I_{t,c}). The same frame-zero derivative is reused for all intervals. This is a direct one-step, fixed-appearance approximation. It does not differentiate the complete sliding-window training pipeline, its grid interpolation/advection, inflow, or evolving Gaussian appearance. All weights still start at zero when nonlinear fitting begins; fitted surrogate coefficients are not transferred.

At the sixteen sensor locations, S_j evaluates K_j directly. Images and sensors are normalized by the square root of their row counts. Joint OLS selects a block set J to greedily reduce

    min_W ||A_J W - Y_image||_F^2 + ||S_J W - Y_sensor||_F^2.

Image-only OLS omits the second term. These row normalizations do not whiten noise or make the two terms equally informative. Their physical units differ. In the recorded data the initial normalized target energies are 0.000622 for images and 0.738 for sensors. Joint selection initially responds mostly to the sensor term; after fitting it nearly exactly, it can continue fitting weak image directions. This is a specified heuristic proxy, not a calibrated likelihood or the exact upstream nonlinear objective.

## Block selection and regularization

At each step, all remaining candidate blocks and the targets are projected orthogonally away from the selected span. The next block maximizes the reduction in summed squared residuals over the three interval targets. Entire three-column blocks are selected together, preserving the vector kernel. Reorthogonalization and a numerical rank threshold handle redundant blocks. Tests compare greedy choices with exhaustive least-squares refits on small dictionaries, including zero/repeated blocks and coefficient-coordinate rotations/scalings.

Regularized OLS instead selects through the augmented system

    [A_J; S_J; sqrt(lambda) I] W ≈ [Y_image; Y_sensor; 0].

It therefore minimizes the local residual plus lambda ||W||_F^2. Four-fold sensor-location CV chooses lambda through the full candidate dictionary; all times and components at a sensor location stay together. This prior depends on the documented kernel normalization, and it is not interchangeable with a physical field-energy norm. The full-dictionary CV optimum also need not be optimal after selecting a small subset or under the final nonlinear renderer. These are explicit limits of this follow-up.

Separate support-scale CV tests the conventional domain-coverage centers with common radius multipliers 0.5, 1 and 2. It selects multiplier 1 at each tested budget. The identical initializers reuse their existing reconstruction fits, rather than counting duplicate fits as extra evidence.

## Numerical checks and their boundary

The original central-difference image matrix failed an independent renderer derivative check; the tested step sizes did not converge reliably. Its six reconstruction fits remain labelled pilots. The corrected matrix composes the upstream backward operator with the analytic RBF matrix. Four random-probe checks verify assembly against independent vector-Jacobian products, but do not establish that the backward operator is the mathematical derivative of the entire discrete renderer. That unresolved limitation is retained in the results.

`diagnose_surrogate.py` independently solves each selected linear system at budgets 32, 64 and 128. It reports image, sensor and coefficient-prior losses separately, along with singular-value condition numbers. Agreement with the greedy residual traces checks the implementation on the actual recorded dictionaries. The independently fitted surrogate coefficients are never transferred to reconstruction. Consequently, large surrogate coefficients or condition numbers diagnose a weakness of the selection problem; they do not establish the cause of downstream motion errors. A lower proxy loss, better conditioning, or analytically zero divergence is not evidence of better held-out motion; the saved full reconstruction experiments provide that test.

## Attribution and scope

- [Ni et al. (2025), Representing Flow Fields with Divergence-Free Kernels for Reconstruction](https://arxiv.org/abs/2504.01913) supplies the modern kernel reconstruction comparison and the representation used by the upstream project.
- [Tao et al. (2026), LagrangianSplats](https://arxiv.org/abs/2605.09299) already transports Gaussian primitives with divergence-free kernel velocities. Its pinned official implementation supplies the nonlinear reconstruction pipeline here.
- [Chen, Cowan and Grant (1991), Orthogonal least squares learning algorithm for radial basis function networks](https://doi.org/10.1109/72.80341), and [Chen, Chng and Alkadhimi (1996), Regularized orthogonal least squares algorithm for constructing radial basis function networks](https://doi.org/10.1080/00207179608921659), establish the earlier selection/regularization lineage. The implementation here uses the explicitly stated block/coefficient-ridge objective; it is not a reproduction of every detail of those algorithms. The thesis itself attributes OLS to Chen. No first-in-literature claim is made for this initialization experiment.

The broad-domain control changes both center placement and support schedule relative to image-support coverage. Without a factorial ablation, a difference between them cannot be attributed solely to broad support or solely to spatial coverage. The exact-budget comparison identifies which tested initializer works better on this case, not a universal kernel-selection law.
