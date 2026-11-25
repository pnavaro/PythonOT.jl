@testset "da" begin
    # test_da.py::test_sinkhorn_l1l2_transport_class / test_emd_laplace
    n = 10
    xs = [Float64(i) for i in 1:n, _ in 1:2]
    xt = xs .+ 0.1
    labels = repeat([1, 2]; inner=n ÷ 2)
    a = fill(1 / n, n)
    b = fill(1 / n, n)
    M = [sum(abs2, xs[i, :] - xt[j, :]) for i in 1:n, j in 1:n]
    M ./= maximum(M)

    G = sinkhorn_l1l2_gl(a, labels, b, M, 1.0; eta=1.0)
    @test vec(sum(G; dims=2)) ≈ a atol = 1e-3
    G = sinkhorn_lpl1_mm(a, labels, b, M, 1.0; eta=1.0)
    @test vec(sum(G; dims=2)) ≈ a atol = 1e-3
    # emd_laplace needs scikit-learn, which is not a dependency of POT
end
