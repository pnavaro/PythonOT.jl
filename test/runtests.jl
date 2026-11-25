using Pkg

# Activate test environment on older Julia versions
@static if VERSION < v"1.2"
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path=dirname(@__DIR__)))
    Pkg.instantiate()
end

using PythonOT
using PythonOT: dist, unif, solve, get_random_projections
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

    @testset "wrappers" begin
        μ = [0.5, 0.2, 0.3]
        ν = [0.4, 0.6]
        C = [0.0 1.0; 2.0 0.0; 0.5 1.5]
        @test partial_wasserstein(μ, ν, C; m=0.5) isa Matrix
        @test size(partial_wasserstein(μ, ν, C; m=0.5)) == (3, 2)
        @test partial_wasserstein2(μ, ν, C; m=0.5) isa Real
        @test sum(partial_wasserstein(μ, ν, C; m=0.5)) ≈ 0.5
        @test sinkhorn_log(μ, ν, C, 0.1) ≈ sinkhorn(μ, ν, C, 0.1) atol = 1e-6
        @test sinkhorn_stabilized(μ, ν, C, 0.1) ≈ sinkhorn(μ, ν, C, 0.1) atol = 1e-6
        @test sum(greenkhorn(μ, ν, C, 0.1)) ≈ 1 atol = 1e-4
        @test unif(4) == fill(0.25, 4)
        D = dist([0.0 0.0; 1.0 1.0], [0.0 0.0; 2.0 0.0])
        @test D == [0.0 4.0; 2.0 2.0]
        @test wasserstein_1d([0.0, 1.0], [1.0, 2.0]) ≈ 1
        @test mm_unbalanced2(μ[1:2], ν, C[1:2, :], 5) isa Real
        @test lbfgsb_unbalanced(μ, ν, C, 0.1, 1.0) isa Matrix
        @test size(
            empirical_sinkhorn([0.0 0.0; 1.0 1.0], [0.0 1.0; 2.0 2.0; 3.0 3.0], 1.0)
        ) == (2, 3)
        @test empirical_sinkhorn2([0.0; 1.0;;], [2.0; 3.0;;], 0.1) isa Any

        xs = [0.0 0.0; 1.0 0.0; 0.0 1.0]
        Cs = dist(xs, xs)
        p = unif(3)
        @test gromov_wasserstein2(p, p, Cs, Cs) ≈ 0 atol = 1e-8
        @test fused_gromov_wasserstein2(p, p, zeros(3, 3), Cs, Cs) ≈ 0 atol = 1e-8
        @test size(fused_gromov_wasserstein(p, p, zeros(3, 3), Cs, Cs)) == (3, 3)
        @test entropic_gromov_wasserstein2(p, p, Cs, Cs, 0.1) isa Real
        @test size(semirelaxed_gromov_wasserstein(Cs, Cs, p)) == (3, 3)
        @test size(partial_gromov_wasserstein(Cs, Cs, p, p; m=0.5)) == (3, 3)
        @test partial_gromov_wasserstein2(Cs, Cs, p, p; m=0.5) isa Real
        @test size(entropic_partial_gromov_wasserstein(Cs, Cs, p, p, 1.0; m=0.5)) == (3, 3)
        @test entropic_partial_gromov_wasserstein2(Cs, Cs, p, p, 1.0; m=0.5) isa Real
        @test size(gromov_barycenters(3, [Cs, Cs]; max_iter=5)) == (3, 3)

        X = [0.0 0.0; 1.0 1.0; 2.0 0.5]
        @test sliced_wasserstein_distance(X, X; seed=0) ≈ 0 atol = 1e-8
        @test max_sliced_wasserstein_distance(X, X; seed=0) ≈ 0 atol = 1e-8
        @test size(get_random_projections(3, 5; seed=0)) == (3, 5)
        @test only(wasserstein_circle([0.1, 0.5], [0.1, 0.5])) ≈ 0 atol = 1e-8

        res = solve(C, μ, ν)
        @test res.value isa Real
    end
end
