@testset "smooth" begin
    using PythonOT.Smooth
    # test_smooth.py: the dual and semi-dual solutions satisfy the marginals
    a = [0.5, 0.2, 0.3]
    b = [0.4, 0.6]
    M = [0.0 1.0; 2.0 0.0; 0.5 1.5]
    for reg_type in ("l2", "kl")
        G = smooth_ot_dual(a, b, M, 1.0; reg_type=reg_type)
        @test vec(sum(G; dims=2)) ≈ a atol = 1e-4
        @test vec(sum(G; dims=1)) ≈ b atol = 1e-4
        Gs = smooth_ot_semi_dual(a, b, M, 1.0; reg_type=reg_type)
        @test vec(sum(Gs; dims=2)) ≈ a atol = 1e-3
    end
end
