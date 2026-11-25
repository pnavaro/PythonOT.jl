"""
    gromov_wasserstein(μ, ν, Cμ, Cν, loss = "square_loss"; kwargs...)

Compute the exact Gromov-Wasserstein transport plan between `(μ, Cμ)` and `(ν, Cν)`.

The Gromov-Wasserstein transport problem seeks to find a minimizer of 
```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\sum_{i, j, k, l} L((C_μ)_{ik}, (C_ν)_{jl}) \\gamma_{ij} \\gamma_{kl},
```
where ``L`` is quadratic (`loss = "square_loss"`) or the Kullback-Leibler divergence (`loss = "kl_loss"`).

This function is a wrapper of the function
[`gromov_wasserstein`](https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.gromov_wasserstein) in the
Python Optimal Transport package. Keyword arguments are listed in the documentation of the
Python function.
"""
function gromov_wasserstein(μ, ν, Cμ, Cν, loss="square_loss"; kwargs...)
    return pot.gromov.gromov_wasserstein(Cμ, Cν, μ, ν, loss; kwargs...)
end

"""
    entropic_gromov_wasserstein(μ, ν, Cμ, Cν, ε, loss = "square_loss"; kwargs...)

Compute the entropy-regularized Gromov-Wasserstein transport plan between `(μ, Cμ)` and `(ν, Cν)` with parameter `ε`.

The entropy-regularized Gromov-Wasserstein transport problem seeks to find a minimizer of 
```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\sum_{i, j, k, l} L((C_μ)_{ik}, (C_ν)_{jl}) \\gamma_{ij} \\gamma_{kl} + ε \\Omega(\\gamma),
```
where ``L`` is quadratic (`loss = "square_loss"`) or the Kullback-Leibler divergence (`loss = "kl_loss"`)
and ``\\Omega(\\gamma) = \\sum_{ij} \\gamma_{ij} \\log(\\gamma_{ij})`` is the entropic regularization term. 

This function is a wrapper of the function
[`entropic_gromov_wasserstein`](https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.entropic_gromov_wasserstein) in the
Python Optimal Transport package. Keyword arguments are listed in the documentation of the
Python function.
"""
function entropic_gromov_wasserstein(μ, ν, Cμ, Cν, ε, loss="square_loss"; kwargs...)
    return pot.gromov.entropic_gromov_wasserstein(Cμ, Cν, μ, ν, loss, ε; kwargs...)
end

"""
    gromov_wasserstein2(μ, ν, Cμ, Cν, loss="square_loss"; kwargs...)

Compute the exact Gromov-Wasserstein cost between `(μ, Cμ)` and `(ν, Cν)`.

This function is a wrapper of the function
[`ot.gromov.gromov_wasserstein2`](https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.gromov_wasserstein2) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function gromov_wasserstein2(μ, ν, Cμ, Cν, loss="square_loss"; kwargs...)
    return pot.gromov.gromov_wasserstein2(Cμ, Cν, μ, ν, loss; kwargs...)
end

"""
    entropic_gromov_wasserstein2(μ, ν, Cμ, Cν, ε, loss="square_loss"; kwargs...)

Compute the entropy-regularized Gromov-Wasserstein cost between `(μ, Cμ)` and `(ν, Cν)`.

This function is a wrapper of the function
[`ot.gromov.entropic_gromov_wasserstein2`](https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.entropic_gromov_wasserstein2) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function entropic_gromov_wasserstein2(μ, ν, Cμ, Cν, ε, loss="square_loss"; kwargs...)
    return pot.gromov.entropic_gromov_wasserstein2(Cμ, Cν, μ, ν, loss, ε; kwargs...)
end

"""
    fused_gromov_wasserstein(μ, ν, M, Cμ, Cν, loss="square_loss"; kwargs...)

Compute the fused Gromov-Wasserstein transport plan with feature cost `M` (weight `alpha`).

This function is a wrapper of the function
[`ot.gromov.fused_gromov_wasserstein`](https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.fused_gromov_wasserstein) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function fused_gromov_wasserstein(μ, ν, M, Cμ, Cν, loss="square_loss"; kwargs...)
    return pot.gromov.fused_gromov_wasserstein(M, Cμ, Cν, μ, ν, loss; kwargs...)
end

"""
    fused_gromov_wasserstein2(μ, ν, M, Cμ, Cν, loss="square_loss"; kwargs...)

Compute the fused Gromov-Wasserstein cost with feature cost `M` (weight `alpha`).

This function is a wrapper of the function
[`ot.gromov.fused_gromov_wasserstein2`](https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.fused_gromov_wasserstein2) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function fused_gromov_wasserstein2(μ, ν, M, Cμ, Cν, loss="square_loss"; kwargs...)
    return pot.gromov.fused_gromov_wasserstein2(M, Cμ, Cν, μ, ν, loss; kwargs...)
end

"""
    entropic_fused_gromov_wasserstein(μ, ν, M, Cμ, Cν, ε, loss="square_loss"; kwargs...)

Compute the entropy-regularized fused Gromov-Wasserstein transport plan.

This function is a wrapper of the function
[`ot.gromov.entropic_fused_gromov_wasserstein`](https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.entropic_fused_gromov_wasserstein) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function entropic_fused_gromov_wasserstein(μ, ν, M, Cμ, Cν, ε, loss="square_loss"; kwargs...)
    return pot.gromov.entropic_fused_gromov_wasserstein(M, Cμ, Cν, μ, ν, loss, ε; kwargs...)
end

"""
    semirelaxed_gromov_wasserstein(Cμ, Cν, μ, loss="square_loss"; kwargs...)

Compute the semi-relaxed Gromov-Wasserstein transport plan (target marginal is free).

This function is a wrapper of the function
[`ot.gromov.semirelaxed_gromov_wasserstein`](https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.semirelaxed_gromov_wasserstein) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function semirelaxed_gromov_wasserstein(Cμ, Cν, μ, loss="square_loss"; kwargs...)
    return pot.gromov.semirelaxed_gromov_wasserstein(Cμ, Cν, μ, loss; kwargs...)
end

"""
    gromov_barycenters(N, Cs; kwargs...)

Compute the Gromov-Wasserstein barycenter structure matrix of size `N` from the structure matrices `Cs`.

This function is a wrapper of the function
[`ot.gromov.gromov_barycenters`](https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.gromov_barycenters) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function gromov_barycenters(N, Cs; kwargs...)
    return pot.gromov.gromov_barycenters(N, Cs; kwargs...)
end

"""
    fgw_barycenters(N, Ys, Cs; kwargs...)

Compute the fused Gromov-Wasserstein barycenter (features, structure) of size `N`.

This function is a wrapper of the function
[`ot.gromov.fgw_barycenters`](https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.fgw_barycenters) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function fgw_barycenters(N, Ys, Cs; kwargs...)
    return pot.gromov.fgw_barycenters(N, Ys, Cs; kwargs...)
end
