"""
    sinkhorn(μ, ν, C, ε; kwargs...)

Compute the optimal transport plan for the entropic regularization optimal transport problem
with source and target marginals `μ` and `ν`, cost matrix `C` of size
`(length(μ), length(ν))`, and entropic regularization parameter `ε`.

The optimal transport plan `γ` is of the same size as `C` and solves
```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle
+ \\varepsilon \\Omega(\\gamma),
```
where ``\\Omega(\\gamma) = \\sum_{i,j} \\gamma_{i,j} \\log \\gamma_{i,j}`` is the entropic
regularization term.

This function is a wrapper of the function
[`sinkhorn`](https://pythonot.github.io/all.html#ot.sinkhorn) in the Python Optimal
Transport package. Keyword arguments are listed in the documentation of the Python function.

# Examples

```jldoctest sinkhorn
julia> μ = [0.5, 0.2, 0.3];

julia> ν = [0.0, 1.0];

julia> C = [0.0 1.0; 2.0 0.0; 0.5 1.5];

julia> sinkhorn(μ, ν, C, 0.01)
3×2 Matrix{Float64}:
 0.0  0.5
 0.0  0.2
 0.0  0.3
```

It is possible to provide multiple target marginals as columns of a matrix. In this case the
optimal transport costs are returned:

```jldoctest sinkhorn
julia> ν = [0.0 0.5; 1.0 0.5];

julia> round.(sinkhorn(μ, ν, C, 0.01); sigdigits=6)
2-element Vector{Float64}:
 0.95
 0.45
```

See also: [`sinkhorn2`](@ref)
"""
function sinkhorn(μ, ν, C, ε; kwargs...)
    return pot.sinkhorn(μ, ν, PyCall.PyReverseDims(permutedims(C)), ε; kwargs...)
end

"""
    sinkhorn2(μ, ν, C, ε; kwargs...)

Compute the optimal transport cost for the entropic regularization optimal transport problem
with source and target marginals `μ` and `ν`, cost matrix `C` of size
`(length(μ), length(ν))`, and entropic regularization parameter `ε`.

The optimal transport cost is the scalar value
```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle
+ \\varepsilon \\Omega(\\gamma),
```
where ``\\Omega(\\gamma) = \\sum_{i,j} \\gamma_{i,j} \\log \\gamma_{i,j}`` is the entropic
regularization term.

This function is a wrapper of the function
[`sinkhorn2`](https://pythonot.github.io/all.html#ot.sinkhorn2) in the Python Optimal
Transport package. Keyword arguments are listed in the documentation of the Python function.

# Examples

```jldoctest sinkhorn2
julia> μ = [0.5, 0.2, 0.3];

julia> ν = [0.0, 1.0];

julia> C = [0.0 1.0; 2.0 0.0; 0.5 1.5];

julia> round(sinkhorn2(μ, ν, C, 0.01); sigdigits=6)
0.95
```

It is possible to provide multiple target marginals as columns of a matrix.

```jldoctest sinkhorn2
julia> ν = [0.0 0.5; 1.0 0.5];

julia> round.(sinkhorn2(μ, ν, C, 0.01); sigdigits=6)
2-element Vector{Float64}:
 0.95
 0.45
```

See also: [`sinkhorn`](@ref)
"""
function sinkhorn2(μ, ν, C, ε; kwargs...)
    return pot.sinkhorn2(μ, ν, PyCall.PyReverseDims(permutedims(C)), ε; kwargs...)
end

"""
    empirical_sinkhorn_divergence(xsource, xtarget, ε; kwargs...)

Compute the Sinkhorn divergence from empirical data, where `xsource` and `xtarget` are
arrays representing samples in the source domain and target domain, respectively, and  `ε`
is the regularization term.

This function is a wrapper of the function
[`ot.bregman.empirical_sinkhorn_divergence`](https://pythonot.github.io/gen_modules/ot.bregman.html#ot.bregman.empirical_sinkhorn_divergence)
in the Python Optimal Transport package. Keyword arguments are listed in the documentation of the Python function.

# Examples

```jldoctest
julia> xsource = [1.0];

julia> xtarget = [2.0, 3.0];

julia> ε = 0.01;

julia> empirical_sinkhorn_divergence(xsource, xtarget, ε) ≈
       sinkhorn2([1], [0.5, 0.5], [1.0 4.0], ε) -
       (
           sinkhorn2([1], [1], zeros(1, 1), ε) +
           sinkhorn2([0.5, 0.5], [0.5, 0.5], Float64[0 1; 1 0], ε)
       ) / 2
true
```

See also: [`sinkhorn2`](@ref)
"""
function empirical_sinkhorn_divergence(xsource, xtarget, ε; kwargs...)
    return pot.bregman.empirical_sinkhorn_divergence(
        reshape(xsource, Val(2)), reshape(xtarget, Val(2)), ε; kwargs...
    )
end

"""
    barycenter(A, C, ε; kwargs...)

Compute the entropically regularized Wasserstein barycenter with histograms `A`, cost matrix
`C`, and entropic regularization parameter `ε`.

The Wasserstein barycenter is a histogram and solves
```math
\\inf_{a} \\sum_{i} W_{\\varepsilon,C}(a, a_i),
```
where the histograms ``a_i`` are columns of matrix `A` and ``W_{\\varepsilon,C}(a, a_i)``
is the optimal transport cost for the entropically regularized optimal transport problem
with marginals ``a`` and ``a_i``, cost matrix ``C``, and entropic regularization parameter
``\\varepsilon``. Optionally, weights of the histograms ``a_i`` can be provided with the
keyword argument `weights`.

This function is a wrapper of the function
[`barycenter`](https://pythonot.github.io/all.html#ot.barycenter) in the
Python Optimal Transport package. Keyword arguments are listed in the documentation of the
Python function.

# Examples

```jldoctest
julia> A = rand(10, 3);

julia> A ./= sum(A; dims=1);

julia> C = rand(10, 10);

julia> isapprox(sum(barycenter(A, C, 0.01; method="sinkhorn_stabilized")), 1; atol=1e-4)
true
```
"""
barycenter(A, C, ε; kwargs...) = pot.barycenter(A, C, ε; kwargs...)

"""
    sinkhorn_log(μ, ν, C, ε; kwargs...)

Solve entropic regularized OT with the log-domain Sinkhorn algorithm and return the plan.

This function is a wrapper of the function
[`ot.bregman.sinkhorn_log`](https://pythonot.github.io/gen_modules/ot.bregman.html#ot.bregman.sinkhorn_log) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function sinkhorn_log(μ, ν, C, ε; kwargs...)
    return pot.bregman.sinkhorn_log(μ, ν, C, ε; kwargs...)
end

"""
    sinkhorn_stabilized(μ, ν, C, ε; kwargs...)

Solve entropic regularized OT with the stabilized Sinkhorn algorithm and return the plan.

This function is a wrapper of the function
[`ot.bregman.sinkhorn_stabilized`](https://pythonot.github.io/gen_modules/ot.bregman.html#ot.bregman.sinkhorn_stabilized) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function sinkhorn_stabilized(μ, ν, C, ε; kwargs...)
    return pot.bregman.sinkhorn_stabilized(μ, ν, C, ε; kwargs...)
end

"""
    sinkhorn_epsilon_scaling(μ, ν, C, ε; kwargs...)

Solve entropic regularized OT with Sinkhorn and epsilon scaling and return the plan.

This function is a wrapper of the function
[`ot.bregman.sinkhorn_epsilon_scaling`](https://pythonot.github.io/gen_modules/ot.bregman.html#ot.bregman.sinkhorn_epsilon_scaling) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function sinkhorn_epsilon_scaling(μ, ν, C, ε; kwargs...)
    return pot.bregman.sinkhorn_epsilon_scaling(μ, ν, C, ε; kwargs...)
end

"""
    greenkhorn(μ, ν, C, ε; kwargs...)

Solve entropic regularized OT with the Greenkhorn algorithm and return the plan.

This function is a wrapper of the function
[`ot.bregman.greenkhorn`](https://pythonot.github.io/gen_modules/ot.bregman.html#ot.bregman.greenkhorn) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function greenkhorn(μ, ν, C, ε; kwargs...)
    return pot.bregman.greenkhorn(μ, ν, C, ε; kwargs...)
end

"""
    screenkhorn(μ, ν, C, ε; kwargs...)

Solve entropic regularized OT with the Screenkhorn screening algorithm and return the plan.

This function is a wrapper of the function
[`ot.bregman.screenkhorn`](https://pythonot.github.io/gen_modules/ot.bregman.html#ot.bregman.screenkhorn) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function screenkhorn(μ, ν, C, ε; kwargs...)
    return pot.bregman.screenkhorn(μ, ν, C, ε; kwargs...)
end

"""
    empirical_sinkhorn(xsource, xtarget, ε; kwargs...)

Solve entropic regularized OT between empirical samples (rows are samples) and return the plan.

This function is a wrapper of the function
[`ot.bregman.empirical_sinkhorn`](https://pythonot.github.io/gen_modules/ot.bregman.html#ot.bregman.empirical_sinkhorn) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function empirical_sinkhorn(xsource, xtarget, ε; kwargs...)
    return pot.bregman.empirical_sinkhorn(xsource, xtarget, ε; kwargs...)
end

"""
    empirical_sinkhorn2(xsource, xtarget, ε; kwargs...)

Solve entropic regularized OT between empirical samples (rows are samples) and return the cost.

This function is a wrapper of the function
[`ot.bregman.empirical_sinkhorn2`](https://pythonot.github.io/gen_modules/ot.bregman.html#ot.bregman.empirical_sinkhorn2) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function empirical_sinkhorn2(xsource, xtarget, ε; kwargs...)
    return pot.bregman.empirical_sinkhorn2(xsource, xtarget, ε; kwargs...)
end

"""
    barycenter_debiased(A, C, ε; kwargs...)

Compute the debiased entropically regularized Wasserstein barycenter of histograms `A` (columns).

This function is a wrapper of the function
[`ot.bregman.barycenter_debiased`](https://pythonot.github.io/gen_modules/ot.bregman.html#ot.bregman.barycenter_debiased) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function barycenter_debiased(A, C, ε; kwargs...)
    return pot.bregman.barycenter_debiased(A, C, ε; kwargs...)
end

"""
    convolutional_barycenter2d(A, ε; kwargs...)

Compute the entropic Wasserstein barycenter of 2D images stacked in the 3D array `A` with a convolutional Sinkhorn scheme.

This function is a wrapper of the function
[`ot.bregman.convolutional_barycenter2d`](https://pythonot.github.io/gen_modules/ot.bregman.html#ot.bregman.convolutional_barycenter2d) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function convolutional_barycenter2d(A, ε; kwargs...)
    return pot.bregman.convolutional_barycenter2d(A, ε; kwargs...)
end
