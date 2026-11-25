# API

## Exact optimal transport

```@docs
emd
emd2
emd_1d
emd2_1d
wasserstein_1d
wasserstein_circle
binary_search_circle
linear_circular_ot
free_support_barycenter
barycenter_debiased
convolutional_barycenter2d
sinkhorn_log
sinkhorn_stabilized
sinkhorn_epsilon_scaling
greenkhorn
screenkhorn
empirical_sinkhorn
empirical_sinkhorn2
```

## Regularized optimal transport

```@docs
sinkhorn
sinkhorn2
empirical_sinkhorn_divergence
barycenter
```

The submodule `Smooth` contains a function for solving regularized optimal
transport problems with L2- and entropic regularization using the dual
formulation. You can load the submodule with
```julia
using PythonOT.Smooth
```

```@docs
PythonOT.Smooth.smooth_ot_dual
PythonOT.Smooth.smooth_ot_semi_dual
```

## Unbalanced optimal transport

```@docs
sinkhorn_unbalanced
sinkhorn_unbalanced2
barycenter_unbalanced
mm_unbalanced
mm_unbalanced2
sinkhorn_unbalanced_translation_invariant
lbfgsb_unbalanced
lbfgsb_unbalanced2
uot_1d
```

## Partial optimal transport

```@docs
entropic_partial_wasserstein
partial_wasserstein
partial_wasserstein2
partial_wasserstein_lagrange
```

## Gromov-Wasserstein

```@docs
gromov_wasserstein
gromov_wasserstein2
entropic_gromov_wasserstein
entropic_gromov_wasserstein2
fused_gromov_wasserstein
fused_gromov_wasserstein2
entropic_fused_gromov_wasserstein
semirelaxed_gromov_wasserstein
partial_gromov_wasserstein
partial_gromov_wasserstein2
entropic_partial_gromov_wasserstein
entropic_partial_gromov_wasserstein2
gromov_barycenters
fgw_barycenters
```

## Sliced optimal transport

```@docs
sliced_wasserstein_distance
max_sliced_wasserstein_distance
sliced_wasserstein_sphere
PythonOT.get_random_projections
```

## Domain adaptation

```@docs
emd_laplace
sinkhorn_l1l2_gl
sinkhorn_lpl1_mm
```

## Utilities and generic solvers

These helpers are not exported; call them as `PythonOT.dist`, etc.

```@docs
PythonOT.dist
PythonOT.unif
PythonOT.solve
PythonOT.solve_sample
```
