module PythonOT

using PyCall: PyCall

export emd,
    emd2,
    emd_1d,
    emd2_1d,
    sinkhorn,
    sinkhorn2,
    barycenter,
    barycenter_unbalanced,
    sinkhorn_unbalanced,
    sinkhorn_unbalanced2,
    empirical_sinkhorn_divergence,
    mm_unbalanced,
    entropic_partial_wasserstein,
    entropic_partial_gromov_wasserstein,
    entropic_partial_gromov_wasserstein2,
    partial_wasserstein,
    partial_wasserstein2,
    gromov_wasserstein,
    entropic_gromov_wasserstein,
    partial_wasserstein_lagrange,
    partial_gromov_wasserstein,
    partial_gromov_wasserstein2,
    wasserstein_1d,
    wasserstein_circle,
    binary_search_circle,
    linear_circular_ot,
    free_support_barycenter,
    sinkhorn_log,
    sinkhorn_stabilized,
    sinkhorn_epsilon_scaling,
    greenkhorn,
    screenkhorn,
    empirical_sinkhorn,
    empirical_sinkhorn2,
    barycenter_debiased,
    convolutional_barycenter2d,
    mm_unbalanced2,
    sinkhorn_unbalanced_translation_invariant,
    lbfgsb_unbalanced,
    lbfgsb_unbalanced2,
    uot_1d,
    gromov_wasserstein2,
    entropic_gromov_wasserstein2,
    fused_gromov_wasserstein,
    fused_gromov_wasserstein2,
    entropic_fused_gromov_wasserstein,
    semirelaxed_gromov_wasserstein,
    gromov_barycenters,
    fgw_barycenters,
    sliced_wasserstein_distance,
    max_sliced_wasserstein_distance,
    sliced_wasserstein_sphere,
    emd_laplace,
    sinkhorn_l1l2_gl,
    sinkhorn_lpl1_mm

const pot = PyCall.PyNULL()

for f in ("lp", "bregman", "unbalanced", "partial", "gromov", "sliced", "da", "utils")
    include(f * ".jl")
end
include("smooth.jl")

function __init__()
    return copy!(pot, PyCall.pyimport_conda("ot", "pot", "conda-forge"))
end

end
