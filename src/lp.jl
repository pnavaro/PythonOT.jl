"""
    emd(μ, ν, C; kwargs...)

Compute the optimal transport plan for the Monge-Kantorovich problem with source and target
marginals `μ` and `ν` and cost matrix `C` of size `(length(μ), length(ν))`.

The optimal transport plan `γ` is of the same size as `C` and solves
```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle.
```

This function is a wrapper of the function
[`emd`](https://pythonot.github.io/all.html#ot.emd) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.

# Examples

```jldoctest
julia> μ = [0.5, 0.2, 0.3];

julia> ν = [0.0, 1.0];

julia> C = [0.0 1.0; 2.0 0.0; 0.5 1.5];

julia> emd(μ, ν, C)
3×2 Matrix{Float64}:
 0.0  0.5
 0.0  0.2
 0.0  0.3
```

See also: [`emd2`](@ref)
"""
function emd(μ, ν, C; kwargs...)
    return pot.lp.emd(μ, ν, PyCall.PyReverseDims(permutedims(C)); kwargs...)
end

"""
    emd2(μ, ν, C; kwargs...)

Compute the optimal transport cost for the Monge-Kantorovich problem with source and target
marginals `μ` and `ν` and cost matrix `C` of size `(length(μ), length(ν))`.

The optimal transport cost is the scalar value
```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle.
```

This function is a wrapper of the function
[`emd2`](https://pythonot.github.io/all.html#ot.emd2) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.

# Examples

```jldoctest
julia> μ = [0.5, 0.2, 0.3];

julia> ν = [0.0, 1.0];

julia> C = [0.0 1.0; 2.0 0.0; 0.5 1.5];

julia> emd2(μ, ν, C)
0.95
```

See also: [`emd`](@ref)
"""
function emd2(μ, ν, C; kwargs...)
    return pot.lp.emd2(μ, ν, PyCall.PyReverseDims(permutedims(C)); kwargs...)
end

"""
    emd_1d(xsource, xtarget; kwargs...)

Compute the optimal transport plan for the Monge-Kantorovich problem with univariate
discrete measures with support `xsource` and `xtarget` as source and target marginals.

This function is a wrapper of the function
[`emd_1d`](https://pythonot.github.io/all.html#ot.emd_1d) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.

# Examples

```jldoctest
julia> xsource = [0.2, 0.5];

julia> xtarget = [0.8, 0.3];

julia> emd_1d(xsource, xtarget)
2×2 Matrix{Float64}:
 0.0  0.5
 0.5  0.0

julia> histogram_source = [0.8, 0.2];

julia> histogram_target = [0.7, 0.3];

julia> emd_1d(xsource, xtarget; a=histogram_source, b=histogram_target)
2×2 Matrix{Float64}:
 0.5  0.3
 0.2  0.0
```

See also: [`emd`](@ref), [`emd2_1d`](@ref)
"""
function emd_1d(xsource, xtarget; kwargs...)
    return pot.lp.emd_1d(xsource, xtarget; kwargs...)
end

"""
    emd2_1d(xsource, xtarget; kwargs...)

Compute the optimal transport cost for the Monge-Kantorovich problem with univariate
discrete measures with support `xsource` and `xtarget` as source and target marginals.

This function is a wrapper of the function
[`emd2_1d`](https://pythonot.github.io/all.html#ot.emd2_1d) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.

# Examples

```jldoctest
julia> xsource = [0.2, 0.5];

julia> xtarget = [0.8, 0.3];

julia> round(emd2_1d(xsource, xtarget); sigdigits=6)
0.05

julia> histogram_source = [0.8, 0.2];

julia> histogram_target = [0.7, 0.3];

julia> round(emd2_1d(xsource, xtarget; a=histogram_source, b=histogram_target); sigdigits=6)
0.201
```

See also: [`emd2`](@ref), [`emd2_1d`](@ref)
"""
function emd2_1d(xsource, xtarget; kwargs...)
    return pot.lp.emd2_1d(xsource, xtarget; kwargs...)
end

"""
    wasserstein_1d(u_values, v_values; kwargs...)

Compute the 1D Wasserstein distance (power `p`) between two weighted empirical measures.

This function is a wrapper of the function
[`ot.lp.wasserstein_1d`](https://pythonot.github.io/gen_modules/ot.lp.html#ot.lp.wasserstein_1d) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function wasserstein_1d(u_values, v_values; kwargs...)
    return pot.lp.wasserstein_1d(u_values, v_values; kwargs...)
end

"""
    wasserstein_circle(u_values, v_values; kwargs...)

Compute the Wasserstein distance on the circle (values in [0, 1)) between two weighted empirical measures.

This function is a wrapper of the function
[`ot.lp.wasserstein_circle`](https://pythonot.github.io/gen_modules/ot.lp.html#ot.lp.wasserstein_circle) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function wasserstein_circle(u_values, v_values; kwargs...)
    return pot.lp.wasserstein_circle(u_values, v_values; kwargs...)
end

"""
    binary_search_circle(u_values, v_values; kwargs...)

Compute the Wasserstein distance on the circle using a binary search.

This function is a wrapper of the function
[`ot.lp.binary_search_circle`](https://pythonot.github.io/gen_modules/ot.lp.html#ot.lp.binary_search_circle) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function binary_search_circle(u_values, v_values; kwargs...)
    return pot.lp.binary_search_circle(u_values, v_values; kwargs...)
end

"""
    linear_circular_ot(u_values, v_values=nothing; kwargs...)

Compute the linear circular OT embedding / distance between measures on the circle.

This function is a wrapper of the function
[`ot.lp.linear_circular_ot`](https://pythonot.github.io/gen_modules/ot.lp.html#ot.lp.linear_circular_ot) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function linear_circular_ot(u_values, v_values=nothing; kwargs...)
    return pot.lp.linear_circular_ot(u_values, v_values; kwargs...)
end

"""
    free_support_barycenter(measures_locations, measures_weights, X_init; kwargs...)

Compute the Wasserstein barycenter of measures with free support, starting from `X_init`.

This function is a wrapper of the function
[`ot.lp.free_support_barycenter`](https://pythonot.github.io/gen_modules/ot.lp.html#ot.lp.free_support_barycenter) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function free_support_barycenter(measures_locations, measures_weights, X_init; kwargs...)
    return pot.lp.free_support_barycenter(measures_locations, measures_weights, X_init; kwargs...)
end
