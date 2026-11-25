"""
    emd_laplace(a, b, xs, xt, M; kwargs...)

Solve the optimal transport problem with Laplacian regularization (domain adaptation).

This function is a wrapper of the function
[`ot.da.emd_laplace`](https://pythonot.github.io/gen_modules/ot.da.html#ot.da.emd_laplace) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function emd_laplace(a, b, xs, xt, M; kwargs...)
    return pot.da.emd_laplace(a, b, xs, xt, M; kwargs...)
end

"""
    sinkhorn_l1l2_gl(a, labels_a, b, M, reg; kwargs...)

Solve entropic OT with a group-lasso (l1-l2) class regularizer (domain adaptation).

This function is a wrapper of the function
[`ot.da.sinkhorn_l1l2_gl`](https://pythonot.github.io/gen_modules/ot.da.html#ot.da.sinkhorn_l1l2_gl) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function sinkhorn_l1l2_gl(a, labels_a, b, M, reg; kwargs...)
    return pot.da.sinkhorn_l1l2_gl(a, labels_a, b, M, reg; kwargs...)
end

"""
    sinkhorn_lpl1_mm(a, labels_a, b, M, reg; kwargs...)

Solve entropic OT with an lp-l1 class regularizer (domain adaptation).

This function is a wrapper of the function
[`ot.da.sinkhorn_lpl1_mm`](https://pythonot.github.io/gen_modules/ot.da.html#ot.da.sinkhorn_lpl1_mm) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function sinkhorn_lpl1_mm(a, labels_a, b, M, reg; kwargs...)
    return pot.da.sinkhorn_lpl1_mm(a, labels_a, b, M, reg; kwargs...)
end
