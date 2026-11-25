"""
    entropic_partial_wasserstein(a, b, M, reg; kwargs...)

Solves the partial optimal transport problem and returns the OT plan
The function considers the following problem:

```math
\\gamma = \\mathop{\\arg \\min}_\\gamma \\quad \\langle \\gamma,
\\mathbf{M} \\rangle_F + \\mathrm{reg} \\cdot\\Omega(\\gamma)
```

- `a` and `b` are the sample weights
- `M` is the metric cost matrix
- `reg` is a regularization term > 0

This function is a wrapper of the function
[`entropic_partial_wasserstein`](https://pythonot.github.io/gen_modules/ot.partial.html#ot.partial.entropic_partial_wasserstein) in the
Python Optimal Transport package. Keyword arguments are listed in the documentation of the
Python function.


# Examples

```jldoctest
julia> a = [.1, .2];

julia> b = [.1, .1];

julia> M = [0. 1.; 2. 3.];

julia> round.(entropic_partial_wasserstein(a, b, M, 1, m=0.1), digits=2)
2×2 Matrix{Float64}:
 0.06  0.02
 0.01  0.0
```

"""
function entropic_partial_wasserstein(a, b, M, reg; kwargs...)
    return pot.partial.entropic_partial_wasserstein(a, b, M, reg; kwargs...)
end

"""
    partial_wasserstein(a, b, M; kwargs...)

Solve the partial optimal transport problem and return the OT plan.

This function is a wrapper of the function
[`ot.partial.partial_wasserstein`](https://pythonot.github.io/gen_modules/ot.partial.html#ot.partial.partial_wasserstein) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function partial_wasserstein(a, b, M; kwargs...)
    return pot.partial.partial_wasserstein(a, b, M; kwargs...)
end

"""
    partial_wasserstein2(a, b, M; kwargs...)

Solve the partial optimal transport problem and return the OT cost.

This function is a wrapper of the function
[`ot.partial.partial_wasserstein2`](https://pythonot.github.io/gen_modules/ot.partial.html#ot.partial.partial_wasserstein2) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function partial_wasserstein2(a, b, M; kwargs...)
    return pot.partial.partial_wasserstein2(a, b, M; kwargs...)
end

"""
    partial_wasserstein_lagrange(a, b, M; kwargs...)

Solve the partial optimal transport problem with a Lagrangian relaxation (`reg_m`) and return the OT plan.

This function is a wrapper of the function
[`ot.partial.partial_wasserstein_lagrange`](https://pythonot.github.io/gen_modules/ot.partial.html#ot.partial.partial_wasserstein_lagrange) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function partial_wasserstein_lagrange(a, b, M; kwargs...)
    return pot.partial.partial_wasserstein_lagrange(a, b, M; kwargs...)
end
