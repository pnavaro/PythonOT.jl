"""
    dist(x1, x2=nothing; kwargs...)

Compute the pairwise distance matrix between the rows of `x1` and `x2` (default `x1`); use `metric` to select the metric.

This function is a wrapper of the function
[`ot.utils.dist`](https://pythonot.github.io/gen_modules/ot.utils.html#ot.utils.dist) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function dist(x1, x2=nothing; kwargs...)
    return pot.utils.dist(x1, x2; kwargs...)
end

"""
    unif(n)

Return the uniform histogram of length `n`.

This function is a wrapper of the function
[`ot.utils.unif`](https://pythonot.github.io/gen_modules/ot.utils.html#ot.utils.unif) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function unif(n)
    return pot.utils.unif(n)
end

"""
    solve(M, a=nothing, b=nothing; kwargs...)

Generic OT solver for cost matrix `M`; returns a Python `OTResult` object.

This function is a wrapper of the function
[`ot.solve`](https://pythonot.github.io/gen_modules/all.html#ot.solve) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function solve(M, a=nothing, b=nothing; kwargs...)
    return pot.solve(M, a, b; kwargs...)
end

"""
    solve_sample(xsource, xtarget, a=nothing, b=nothing; kwargs...)

Generic OT solver for empirical samples; returns a Python `OTResult` object.

This function is a wrapper of the function
[`ot.solve_sample`](https://pythonot.github.io/gen_modules/all.html#ot.solve_sample) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function solve_sample(xsource, xtarget, a=nothing, b=nothing; kwargs...)
    return pot.solve_sample(xsource, xtarget, a, b; kwargs...)
end
