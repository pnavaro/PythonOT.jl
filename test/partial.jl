@testset "partial" begin
    # test_partial.py::test_partial_wasserstein
    a = [0.5, 0.2, 0.3]
    b = [0.4, 0.6]
    M = [0.0 1.0; 2.0 0.0; 0.5 1.5]
    G = partial_wasserstein(a, b, M; m=0.5)
    @test sum(G) ≈ 0.5
    @test all(vec(sum(G; dims=2)) .≤ a .+ 1e-8)
    @test all(vec(sum(G; dims=1)) .≤ b .+ 1e-8)
    @test partial_wasserstein2(a, b, M; m=0.5) ≈ sum(G .* M)
    Gl = partial_wasserstein_lagrange(a, b, M; reg_m=1.0)
    @test size(Gl) == (3, 2)
    Ge = entropic_partial_wasserstein(a, b, M, 0.1; m=0.5)
    @test sum(Ge) ≈ 0.5 atol = 1e-3
end
