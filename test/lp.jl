@testset "lp" begin
    # ot.lp.emd_1d / wasserstein_1d (test_1d_solver.py)
    x = [0.0, 1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0, 4.0]
    @test wasserstein_1d(x, y; p=1) ≈ 1
    @test wasserstein_1d(x, y; p=2) ≈ 1
    @test wasserstein_1d(x, x) ≈ 0 atol = 1e-12

    # circle distances (test_1d_solver.py)
    u = [0.1, 0.4, 0.7]
    v = [0.2, 0.5, 0.9]
    w = only(wasserstein_circle(u, v))
    @test w ≥ 0
    @test only(wasserstein_circle(u, u)) ≈ 0 atol = 1e-8
    @test only(first(binary_search_circle(u, v; p=2))) ≈ only(wasserstein_circle(u, v; p=2)) atol =
        1e-2
    @test only(wasserstein_circle(u, u; p=2)) ≈ 0 atol = 1e-8
    @test only(linear_circular_ot(u, v)) ≥ 0

    # free support barycenter (test_ot.py::test_free_support_barycenter)
    measures_locations = [[0.0; 1.0;;], [1.0; 2.0;;]]
    measures_weights = [[0.5, 0.5], [0.5, 0.5]]
    X_init = [0.5; 1.5;;]
    X = free_support_barycenter(measures_locations, measures_weights, X_init)
    @test size(X) == (2, 1)
    @test sort(vec(X)) ≈ [0.5, 1.5] atol = 1e-6
end
