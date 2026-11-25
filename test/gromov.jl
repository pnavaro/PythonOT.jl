@testset "gromov" begin
    # test_gromov.py: identical structures have zero cost
    xs = [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0]
    C1 = [sqrt(sum(abs2, xs[i, :] - xs[j, :])) for i in 1:4, j in 1:4]
    p = fill(0.25, 4)
    @test gromov_wasserstein2(p, p, C1, C1) ≥ 0
    G = gromov_wasserstein(p, p, C1, C1)
    @test vec(sum(G; dims=2)) ≈ p atol = 1e-8
    @test vec(sum(G; dims=1)) ≈ p atol = 1e-8
    Ge = entropic_gromov_wasserstein(p, p, C1, C1, 0.1)
    @test vec(sum(Ge; dims=2)) ≈ p atol = 1e-5
    @test entropic_gromov_wasserstein2(p, p, C1, C1, 0.1) isa Real

    # fused GW with a zero feature cost reduces to GW
    M = zeros(4, 4)
    @test fused_gromov_wasserstein2(p, p, M, C1, C1) ≥ 0
    Gf = fused_gromov_wasserstein(p, p, M, C1, C1)
    @test vec(sum(Gf; dims=2)) ≈ p atol = 1e-8
    Gef = entropic_fused_gromov_wasserstein(p, p, M, C1, C1, 0.1)
    @test vec(sum(Gef; dims=2)) ≈ p atol = 1e-5

    # semi-relaxed GW keeps the source marginal
    Gs = semirelaxed_gromov_wasserstein(C1, C1, p)
    @test vec(sum(Gs; dims=2)) ≈ p atol = 1e-8

    # barycenters (test_gromov.py::test_gromov_barycenter, test_fgw_barycenter)
    Cb = gromov_barycenters(4, [C1, C1]; max_iter=20)
    @test size(Cb) == (4, 4)
    Ys = [xs[:, 1:1], xs[:, 1:1]]
    X, Cf = fgw_barycenters(4, Ys, [C1, C1]; max_iter=5)
    @test size(X) == (4, 1)
    @test size(Cf) == (4, 4)

    # partial GW (test_partial.py::test_partial_gromov_wasserstein)
    m = 0.5
    Gp = partial_gromov_wasserstein(C1, C1, p, p; m=m)
    @test sum(Gp) ≈ m atol = 1e-6
    @test partial_gromov_wasserstein2(C1, C1, p, p; m=m) isa Real
    Gep = entropic_partial_gromov_wasserstein(C1, C1, p, p, 10.0; m=m)
    @test sum(Gep) ≈ m atol = 1e-3
    @test entropic_partial_gromov_wasserstein2(C1, C1, p, p, 10.0; m=m) isa Real
end
