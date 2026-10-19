# Mathematical and statistical specification

## Vector kernels and regularization

Let r=(x−z)/ell, p=exp(−||r||²/2), and d be the vector/spatial dimension. The implementation uses

- K_df = p [((d−1−||r||²) I + rrᵀ)/(d−1)],
- K_cf = p [I−rrᵀ],
- K_ind = p I,
- K_mix = (1−eta) K_df + eta K_cf, 0≤eta≤1.

Each has K(x,x)=I. The differential construction is established mathematics; normalization removes the scalar length-dependent amplitude. The mixed model uses one common length, unlike more flexible published Helmholtz Gaussian processes. The independent Gaussian kernel is not itself a member of this fixed-potential mixture family: canceling the rrᵀ terms still leaves a distance-dependent scalar factor. Thus the tested mixture does not contain every baseline, and nested model selection need not improve test error.

Rows/columns use point-major component ordering. After fitting a component-wise training mean b, solve (K+lambda I) alpha = y−b with Cholesky. The internal common scalar standardization cancels from the linear predictor. This solves the vector RKHS objective ||y−b−K alpha||² + lambda alphaᵀ K alpha for fixed b (with the usual interpretation on the kernel range); lambda is not an index-neighbor finite-difference penalty. b is an empirical training mean, not a jointly optimized universal-kriging mean in this base model. No statistical predictive covariance is calculated.

Every candidate's mean comes only from its training fold. Hyperparameters minimize aggregate validation SSE divided by number of validation vectors. Selected CV loss is optimistic for its own selection task and is not presented as unbiased performance. Outer public-data folds remain excluded from tuning.

## Why the dipole slice violates a planar constraint

For a z-directed unit point dipole,

B(x)=3 x z/||x||⁵ − e_z/||x||³.

Away from the origin, div_3 B=0. On y=0, however,

∂x Bx + ∂z Bz = −3z/(x²+z²)^(5/2).

The 3D and 2D experiments use different domains and numbers of vector components, so their absolute MSEs should not be directly compared. Compare methods within each setting. The analytic identity does not prove which issue affected the original, unavailable MATLAB code.

## Track ambiguity and the finite-basis audit

Take psi=sin(2pi x) sin²(4pi(y−1/8)) and v=(∂y psi,−∂x psi). Both components vanish on every point of the four lines y=1/8,3/8,5/8,7/8, while v is nonzero between them. If v0 fits those observations, v0+a v does as well for every a. No estimator using only those values can identify a without extra assumptions. This is a constructive instance of standard inverse-problem nonidentifiability, not a new general theorem.

For a chosen Fourier streamfunction basis, A maps coefficients c to observations and B maps them to m target vectors. With orthonormal Z spanning ker(A), the exact target ambiguity radius under ||c||≤1 is sigma_max(BZ)/sqrt(m). The coefficient budget is an explicit potential-space prior with decay (1+||k||²)^−1.5. The radius has no unrestricted-field guarantee; a full-rank finite basis can still omit physically important fields.

To avoid treating tiny jitter as complete resolution, also compute

R_delta² = max_c [cᵀ(BᵀB/m)c] / [cᵀ(I+AᵀA/(n delta²))c].

This is a generalized eigenvalue. It is exact for the combined ellipsoid ||c||²+||Ac||²/(n delta²)≤1. That ellipsoid is contained in the intersection of the two separate budgets, so its radius is a **lower bound** on the worst case over that intersection. It is not an upper reconstruction-error bound, posterior standard deviation, calibrated interval, or model of averaging independent noise. Numerical rank uses relative tolerance 1e−10.

## Curved coordinates and variable magnetic field

In the electrostatic planar approximation B=B(x,y)e_z, E=−grad Phi, v=E×B/B² implies div_h(Bv)=0. Ordinary div_h v need not vanish.

Use X(s,n)=(s,n+a sin(2pi s)), J=DX, det J=1, B=exp(beta x). Let T(x)=J(X^−1x)/B(x). A reference solenoidal field u maps to v(x)=T(x)u(X^−1x). The contravariant Piola identity gives div_x(Bv)=0. The kernel is

K_X(x,z)=T(x) K_df(X^−1x,X^−1z) T(z)ᵀ.

This covariance is positive semidefinite by congruence and encodes the weighted physical constraint. A constant reference-flux background is jointly estimated by GLS: with L=K_X+lambda I and H stacking T(x), b=(HᵀL^−1H)^−1HᵀL^−1y; alpha=L^−1(y−Hb). Prediction adds T(q)b. The test uses known synthetic curvature and B, not inferred imagery or realistic geomagnetic gradients. It therefore demonstrates conditional feasibility and sensitivity, not an auroral result.

## Gaussian transport and density ambiguity

For a material point and local deformation gradient, dmu/dt=v(mu), dF/dt=J_v(mu)F. Transport covariance as Sigma(t)=F Sigma0 Fᵀ. This is exact under affine flow and a local linear approximation under nonlinear flow; a generally deformed Gaussian need not remain Gaussian. Jacobi's formula gives d log det F/dt=div v. Incompressibility preserves det F in continuous time. Our ordinary RK4 approximates this, and convergence is measured without projecting determinants to one.

This concerns volume of a local material element. Gaussian opacity, appearance, splitting and merging in a renderer do not automatically obey mass conservation. Time is the trajectory parameter; the relevant constraint is 3D spatial divergence, not divergence of a fictitious four-component spacetime velocity.

For rho(x)=C exp(−xᵀSigma^−1x/2), Sigma=S Sᵀ, take A=S Omega S^−1 for skew-symmetric Omega. Then tr A=0 and A Sigma+Sigma Aᵀ=0. Consequently F=exp(tA) preserves the entire Gaussian density while individual particles move. Unequal diagonal entries of S allow nonzero strain: this is more than an invisible rigid rotation. Every density-only projection with fixed appearance remains identical. This is a single continuous Gaussian density containing unobserved material markers; the plotted finite marker cloud is not itself asserted to be an unchanged finite splat mixture. In this three-parameter affine family, one off-center full-vector velocity observation has rank two and two generic locations have rank three. This does not imply two sensors identify a general flow.

More generally w=grad rho cross grad psi is solenoidal and w·grad rho=0. The public-volume check uses linear psi and commuting finite differences, verifying the instantaneous ambiguity. It does not establish indistinguishable finite-time solutions of the full Navier–Stokes problem with the same boundary conditions. Momentum, texture, forces, boundary observations and multiple independent tracers can supply additional information. The density-only ambiguity is related to the established optical-flow aperture problem.

## Statistical scope

Reported synthetic intervals are 10,000 paired bootstrap resamples of independent field/noise seeds, with fixed generators, grids and selection rules. They are exploratory intervals, without multiplicity correction, and do not establish effects for a population of auroral events. Fluid effects are reported per selected frame and over fixed frames; nearby times are not independent events. Public ocean sampling repeats overlap and cannot be treated as five independent ocean realizations. Numerical divergence and exact constraint tests are diagnostics, not independent accuracy validation.
