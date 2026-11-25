@testset "bregman" begin
    n = 20
    x = collect(range(0, 1; length=n))
    a = fill(1 / n, n)
    b = fill(1 / n, n)
    M = [(xi - xj)^2 for xi in x, xj in x]
    M ./= maximum(M)
    reg = 1.0

    # test_bregman.py: all solvers agree and satisfy the marginals
    G = sinkhorn(a, b, M, reg)
    for solver in (sinkhorn_log, sinkhorn_stabilized)
        Gs = solver(a, b, M, reg)
        @test Gs ≈ G atol = 1e-5
        @test vec(sum(Gs; dims=2)) ≈ a atol = 1e-5
        @test vec(sum(Gs; dims=1)) ≈ b atol = 1e-5
    end
    Ge = sinkhorn_epsilon_scaling(a, b, M, reg)
    @test vec(sum(Ge; dims=2)) ≈ a atol = 1e-4
    Gg = greenkhorn(a, b, M, reg)
    @test vec(sum(Gg; dims=2)) ≈ a atol = 1e-3
    Gscreen = screenkhorn(a, b, M, reg; ns_budget=15, nt_budget=15)
    @test size(Gscreen) == (n, n)

    # empirical Sinkhorn (test_empirical_sinkhorn)
    Xs = reshape(x, n, 1)
    Xt = reshape(x .+ 0.5, n, 1)
    Me = [(xi - xj)^2 for xi in Xs[:, 1], xj in Xt[:, 1]]
    @test empirical_sinkhorn(Xs, Xt, reg) ≈ sinkhorn(a, b, Me, reg) atol = 1e-6
    @test empirical_sinkhorn2(Xs, Xt, reg) ≈ sinkhorn2(a, b, Me, reg) atol = 1e-6
    @test empirical_sinkhorn_divergence(Xs, Xt, reg) isa Real

    # barycenters (test_bregman.py::test_barycenter, test_convolutional_barycenter)
    A = hcat(fill(1 / n, n), [exp(-(xi - 0.5)^2 / 0.02) for xi in x])
    A ./= sum(A; dims=1)
    bary = barycenter_debiased(A, M, reg; weights=[0.5, 0.5])
    @test sum(bary) ≈ 1 atol = 1e-3
    img = zeros(2, 8, 8)
    img[1, 2:3, 2:3] .= 1
    img[2, 6:7, 6:7] .= 1
    img ./= sum(img; dims=(2, 3))
    cb = convolutional_barycenter2d(img, 0.05)
    @test size(cb) == (8, 8)
    @test sum(cb) ≈ 1 atol = 1e-3
end
