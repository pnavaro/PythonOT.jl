"""
    sinkhorn_unbalanced(μ, ν, C, ε, λ; kwargs...)

Compute the optimal transport plan for the unbalanced entropic regularization optimal
transport problem with source and target marginals `μ` and `ν`, cost matrix `C` of size
`(length(μ), length(ν))`, entropic regularization parameter `ε`, and marginal relaxation
term `λ`.

The optimal transport plan `γ` is of the same size as `C` and solves
```math
\\inf_{\\gamma} \\langle \\gamma, C \\rangle
+ \\varepsilon \\Omega(\\gamma)
+ \\lambda \\mathrm{KL}(\\gamma 1, \\mu)
+ \\lambda \\mathrm{KL}(\\gamma^{\\mathsf{T}} 1, \\nu),
```
where ``\\Omega(\\gamma) = \\sum_{i,j} \\gamma_{i,j} \\log \\gamma_{i,j}`` is the entropic
regularization term and ``\\mathrm{KL}`` is the Kullback-Leibler divergence.

This function is a wrapper of the function
[`sinkhorn_unbalanced`](https://pythonot.github.io/all.html#ot.sinkhorn_unbalanced) in the
Python Optimal Transport package. Keyword arguments are listed in the documentation of the
Python function.

# Examples

```jldoctest sinkhorn_unbalanced
julia> μ = [0.5, 0.5];

julia> ν = [0.5, 0.5];

julia> C = [0.0 1.0; 1.0 0.0];

julia> round.(sinkhorn_unbalanced(μ, ν, C, 1, 1); sigdigits=7)
2×2 Matrix{Float64}:
 0.322054  0.118477
 0.118477  0.322054
```

```

See also: [`sinkhorn_unbalanced2`](@ref)
"""
function sinkhorn_unbalanced(μ, ν, C, ε, λ; kwargs...)
    return pot.sinkhorn_unbalanced(
        μ, ν, PyCall.PyReverseDims(permutedims(C)), ε, λ; kwargs...
    )
end

"""
    sinkhorn_unbalanced2(μ, ν, C, ε, λ; kwargs...)

Compute the optimal transport cost for the unbalanced entropic regularization optimal
transport problem with source and target marginals `μ` and `ν`, cost matrix `C` of size
`(length(μ), length(ν))`, entropic regularization parameter `ε`, and marginal relaxation
term `λ`.

The optimal transport cost is the scalar value
```math
\\inf_{\\gamma} \\langle \\gamma, C \\rangle
+ \\varepsilon \\Omega(\\gamma)
+ \\lambda \\mathrm{KL}(\\gamma 1, \\mu)
+ \\lambda \\mathrm{KL}(\\gamma^{\\mathsf{T}} 1, \\nu),
```
where ``\\Omega(\\gamma) = \\sum_{i,j} \\gamma_{i,j} \\log \\gamma_{i,j}`` is the entropic
regularization term and ``\\mathrm{KL}`` is the Kullback-Leibler divergence.

This function is a wrapper of the function
[`sinkhorn_unbalanced2`](https://pythonot.github.io/all.html#ot.sinkhorn_unbalanced2) in the
Python Optimal Transport package. Keyword arguments are listed in the documentation of the
Python function.

# Examples

```jldoctest sinkhorn_unbalanced2
julia> μ = [0.5, 0.1];

julia> ν = [0.5, 0.5];

julia> C = [0.0 1.0; 1.0 0.0];

julia> round.(sinkhorn_unbalanced2(μ, ν, C, 1., 1.); sigdigits=8)
0.19600125
```

See also: [`sinkhorn_unbalanced`](@ref)
"""
function sinkhorn_unbalanced2(μ, ν, C, ε, λ; kwargs...)
    return pot.sinkhorn_unbalanced2(
        μ, ν, PyCall.PyReverseDims(permutedims(C)), ε, λ; kwargs...
    )
end

"""
    barycenter_unbalanced(A, C, ε, λ; kwargs...)

Compute the entropically regularized unbalanced Wasserstein barycenter with histograms `A`, cost matrix
`C`, entropic regularization parameter `ε` and marginal relaxation parameter `λ`.

The Wasserstein barycenter is a histogram and solves
```math
\\inf_{a} \\sum_{i} W_{\\varepsilon,C,\\lambda}(a, a_i),
```
where the histograms ``a_i`` are columns of matrix `A` and ``W_{\\varepsilon,C,\\lambda}(a, a_i)}``
is the optimal transport cost for the entropically regularized optimal transport problem
with marginals ``a`` and ``a_i``, cost matrix ``C``, entropic regularization parameter
``\\varepsilon`` and marginal relaxation parameter ``\\lambda``. Optionally, weights of the histograms ``a_i`` can be provided with the
keyword argument `weights`.

This function is a wrapper of the function
[`barycenter_unbalanced`](https://pythonot.github.io/gen_modules/ot.unbalanced.html#ot.unbalanced.barycenter_unbalanced) in the
Python Optimal Transport package. Keyword arguments are listed in the documentation of the
Python function.

# Examples

```jldoctest
julia> A = rand(10, 3);

julia> A ./= sum(A; dims=1);

julia> C = rand(10, 10);

julia> isapprox(sum(barycenter_unbalanced(A, C, 0.01, 1; method="sinkhorn_stabilized")), 1; atol=1e-4)
false

julia> isapprox(sum(barycenter_unbalanced(
           A, C, 0.01, 10_000; method="sinkhorn_stabilized", numItermax=5_000
       )), 1; atol=1e-4)
true
```

See also: [`barycenter`](@ref)
"""
function barycenter_unbalanced(A, C, ε, λ; kwargs...)
    return pot.barycenter_unbalanced(A, C, ε, λ; kwargs...)
end

"""
    mm_unbalanced(a, b, M, reg_m; reg=0, c=a*b', kwargs...)

Solve the unbalanced optimal transport problem and return the OT plan.
The function solves the following optimization problem:

```math
W = \\min_{\\gamma \\geq 0} \\langle \\gamma, M \\rangle_F +
    \\mathrm{reg_{m1}} \\cdot \\operatorname{div}(\\gamma \\mathbf{1}, a) +
    \\mathrm{reg_{m2}} \\cdot \\operatorname{div}(\\gamma^\\mathsf{T} \\mathbf{1}, b) +
    \\mathrm{reg} \\cdot \\operatorname{div}(\\gamma, c)
```

where

- `M` is the metric cost matrix,
- `a` and `b` are source and target unbalanced distributions,
- `c` is a reference distribution for the regularization,
- `reg_m` is the marginal relaxation term (if it is a scalar or an indexable object of length 1, then the same term is applied to both marginal relaxations), and
- `reg` is a regularization term.

This function is a wrapper of the function
[`mm_unbalanced`](https://pythonot.github.io/gen_modules/ot.unbalanced.html#ot.unbalanced.mm_unbalanced) in the
Python Optimal Transport package. Keyword arguments are listed in the documentation of the
Python function.

# Examples

```jldoctest
julia> a=[.5, .5];

julia> b=[.5, .5];

julia> M=[1. 36.; 9. 4.];

julia> round.(mm_unbalanced(a, b, M, 5, div="kl"), digits=2)
2×2 Matrix{Float64}:
 0.45  0.0
 0.0   0.34

julia> round.(mm_unbalanced(a, b, M, 5, div="l2"), digits=2)
2×2 Matrix{Float64}:
 0.4  0.0
 0.0  0.1
```

"""
function mm_unbalanced(a, b, M, reg_m; kwargs...)
    return pot.unbalanced.mm_unbalanced(a, b, M, reg_m; kwargs...)
end

"""
    mm_unbalanced2(a, b, M, reg_m; kwargs...)

Solve the unbalanced optimal transport problem with the majorization-minimization algorithm and return the cost.

This function is a wrapper of the function
[`ot.unbalanced.mm_unbalanced2`](https://pythonot.github.io/gen_modules/ot.unbalanced.html#ot.unbalanced.mm_unbalanced2) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function mm_unbalanced2(a, b, M, reg_m; kwargs...)
    return pot.unbalanced.mm_unbalanced2(a, b, M, reg_m; kwargs...)
end

"""
    sinkhorn_unbalanced_translation_invariant(a, b, M, reg, reg_m; kwargs...)

Solve the translation-invariant unbalanced entropic OT problem and return the plan.

This function is a wrapper of the function
[`ot.unbalanced.sinkhorn_unbalanced_translation_invariant`](https://pythonot.github.io/gen_modules/ot.unbalanced.html#ot.unbalanced.sinkhorn_unbalanced_translation_invariant) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function sinkhorn_unbalanced_translation_invariant(a, b, M, reg, reg_m; kwargs...)
    return pot.unbalanced.sinkhorn_unbalanced_translation_invariant(a, b, M, reg, reg_m; kwargs...)
end

"""
    lbfgsb_unbalanced(a, b, M, reg, reg_m; kwargs...)

Solve the unbalanced OT problem with L-BFGS-B and return the plan.

This function is a wrapper of the function
[`ot.unbalanced.lbfgsb_unbalanced`](https://pythonot.github.io/gen_modules/ot.unbalanced.html#ot.unbalanced.lbfgsb_unbalanced) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function lbfgsb_unbalanced(a, b, M, reg, reg_m; kwargs...)
    return pot.unbalanced.lbfgsb_unbalanced(a, b, M, reg, reg_m; kwargs...)
end

"""
    lbfgsb_unbalanced2(a, b, M, reg, reg_m; kwargs...)

Solve the unbalanced OT problem with L-BFGS-B and return the cost.

This function is a wrapper of the function
[`ot.unbalanced.lbfgsb_unbalanced2`](https://pythonot.github.io/gen_modules/ot.unbalanced.html#ot.unbalanced.lbfgsb_unbalanced2) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function lbfgsb_unbalanced2(a, b, M, reg, reg_m; kwargs...)
    return pot.unbalanced.lbfgsb_unbalanced2(a, b, M, reg, reg_m; kwargs...)
end

"""
    uot_1d(u_values, v_values, reg_m; kwargs...)

Solve the unbalanced OT problem between 1D measures.

This function is a wrapper of the function
[`ot.unbalanced.uot_1d`](https://pythonot.github.io/gen_modules/ot.unbalanced.html#ot.unbalanced.uot_1d) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function uot_1d(u_values, v_values, reg_m; kwargs...)
    return pot.unbalanced.uot_1d(u_values, v_values, reg_m; kwargs...)
end
