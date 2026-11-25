"""
    sliced_wasserstein_distance(xsource, xtarget; kwargs...)

Compute the sliced Wasserstein distance between empirical samples (rows are samples).

This function is a wrapper of the function
[`ot.sliced.sliced_wasserstein_distance`](https://pythonot.github.io/gen_modules/ot.sliced.html#ot.sliced.sliced_wasserstein_distance) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function sliced_wasserstein_distance(xsource, xtarget; kwargs...)
    return pot.sliced.sliced_wasserstein_distance(xsource, xtarget; kwargs...)
end

"""
    max_sliced_wasserstein_distance(xsource, xtarget; kwargs...)

Compute the max-sliced Wasserstein distance between empirical samples (rows are samples).

This function is a wrapper of the function
[`ot.sliced.max_sliced_wasserstein_distance`](https://pythonot.github.io/gen_modules/ot.sliced.html#ot.sliced.max_sliced_wasserstein_distance) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function max_sliced_wasserstein_distance(xsource, xtarget; kwargs...)
    return pot.sliced.max_sliced_wasserstein_distance(xsource, xtarget; kwargs...)
end

"""
    sliced_wasserstein_sphere(xsource, xtarget; kwargs...)

Compute the spherical sliced Wasserstein distance between samples on the sphere.

This function is a wrapper of the function
[`ot.sliced.sliced_wasserstein_sphere`](https://pythonot.github.io/gen_modules/ot.sliced.html#ot.sliced.sliced_wasserstein_sphere) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function sliced_wasserstein_sphere(xsource, xtarget; kwargs...)
    return pot.sliced.sliced_wasserstein_sphere(xsource, xtarget; kwargs...)
end

"""
    get_random_projections(d, n_projections; kwargs...)

Generate `n_projections` random unit directions in dimension `d` (as columns).

This function is a wrapper of the function
[`ot.sliced.get_random_projections`](https://pythonot.github.io/gen_modules/ot.sliced.html#ot.sliced.get_random_projections) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.
"""
function get_random_projections(d, n_projections; kwargs...)
    return pot.sliced.get_random_projections(d, n_projections; kwargs...)
end
