@testset "utils" begin
    @test unif(4) == fill(0.25, 4)
    @test dist([0.0 0.0; 1.0 1.0], [0.0 0.0; 2.0 0.0]) == [0.0 4.0; 2.0 2.0]
    @test dist([0.0 0.0; 3.0 4.0]; metric="euclidean") ≈ [0.0 5.0; 5.0 0.0]

    # test_solvers.py
    M = [0.0 1.0; 2.0 0.0; 0.5 1.5]
    a = [0.5, 0.2, 0.3]
    b = [0.4, 0.6]
    res = solve(M, a, b)
    @test res.value ≈ emd2(a, b, M)
    @test res.plan ≈ emd(a, b, M)
    xs = [0.0 0.0; 1.0 0.0; 0.0 1.0]
    xt = [0.0 0.0; 2.0 0.0]
    res2 = solve_sample(xs, xt)
    @test size(res2.plan) == (3, 2)
end
