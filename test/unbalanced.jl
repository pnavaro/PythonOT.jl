@testset "unbalanced" begin
    n = 10
    x = collect(range(0, 1; length=n))
    a = fill(1 / n, n)
    b = fill(1 / n, n)
    M = [(xi - xj)^2 for xi in x, xj in x]

    # test_unbalanced.py
    G = mm_unbalanced(a, b, M, 1.0)
    @test size(G) == (n, n)
    @test mm_unbalanced2(a, b, M, 1.0) ≈ sum(G .* M) atol = 1e-6
    Gl = lbfgsb_unbalanced(a, b, M, 0.1, 1.0)
    @test size(Gl) == (n, n)
    @test lbfgsb_unbalanced2(a, b, M, 0.1, 1.0) isa Real
    Gti = sinkhorn_unbalanced_translation_invariant(a, b, M, 1.0, 10.0)
    @test size(Gti) == (n, n)
    @test sum(Gti) ≈ 1 atol = 1e-1
    @test sinkhorn_unbalanced(a, b, M, 1.0, 10.0) isa Matrix

    # test_unbalanced.py::test_uot_1d
    u = [0.0, 1.0, 2.0]
    # the NumPy backend is rejected by POT; only torch and jax are supported
    @test_throws PythonOT.PyCall.PyError uot_1d(u, u, 1.0)
end
