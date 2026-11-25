@testset "sliced" begin
    # test_sliced.py
    X = [0.0 0.0; 1.0 1.0; 2.0 0.5; 0.5 2.0]
    Y = X .+ 1.0
    @test sliced_wasserstein_distance(X, X; seed=0) ≈ 0 atol = 1e-8
    @test sliced_wasserstein_distance(X, Y; seed=0, n_projections=500) > 0
    @test max_sliced_wasserstein_distance(X, X; seed=0) ≈ 0 atol = 1e-8
    @test max_sliced_wasserstein_distance(X, Y; seed=0) > 0
    P = get_random_projections(3, 5; seed=0)
    @test size(P) == (3, 5)
    @test vec(sqrt.(sum(abs2, P; dims=1))) ≈ ones(5) atol = 1e-8
    S = X ./ sqrt.(sum(abs2, X; dims=2))
    S = S[2:end, :]
    @test sliced_wasserstein_sphere(S, S; seed=0) ≈ 0 atol = 1e-6
end
