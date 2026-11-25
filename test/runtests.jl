using Pkg

# Activate test environment on older Julia versions
@static if VERSION < v"1.2"
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path=dirname(@__DIR__)))
    Pkg.instantiate()
end

using PythonOT
using PythonOT: dist, unif, solve, solve_sample, get_random_projections
using Documenter
using Test

# Keep the test output quiet: POT emits deprecation and convergence warnings
PythonOT.PyCall.pyimport("warnings").filterwarnings("ignore")

@testset "PythonOT.jl" begin
    @testset "doctests" begin
        DocMeta.setdocmeta!(PythonOT, :DocTestSetup, :(using PythonOT); recursive=true)
        doctest(
            PythonOT;
            doctestfilters=[
                r"{([a-zA-Z0-9]+,\s?)+[a-zA-Z0-9]+}",
                r"(Array{[a-zA-Z0-9]+,\s?1}|Vector{[a-zA-Z0-9]+})",
                r"(Array{[a-zA-Z0-9]+,\s?2}|Matrix{[a-zA-Z0-9]+})",
            ],
        )
    end

    for f in (
        "lp",
        "bregman",
        "unbalanced",
        "partial",
        "gromov",
        "sliced",
        "da",
        "smooth",
        "utils",
    )
        include(f * ".jl")
    end
end
