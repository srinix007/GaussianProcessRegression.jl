@testset verbose = true "BFGS Quad dim=$dim" for dim = 2:15
    J = rand(dim)
    L = LowerTriangular(rand(dim, dim))
    H = Hermitian(L * L' + 1e-7 * I)
    x0 = rand(dim)
    jac = let J = J, H = H
        x -> J + H * x
    end
    jac1(x) = 3.0 .* ones(dim) + diagm([i * 2.0 for i = 1:dim]) * x
    J1 = 3.0 .* ones(dim)
    H1 = diagm([i * 2.0 for i = 1:dim])
    eps = 1e-3
    max_iter = 100000

    @testset "Hessian finite diff" begin
        @test isposdef(H)
        @test hessian_fd(x -> J + H * x, rand(dim)) ≈ H atol = eps
    end
end

@testset "BFGS hessian update" for dim = 2:30
    L = LowerTriangular(rand(dim, dim))
    B = Hermitian(L * L' + 1e-7 * I)
    s = rand(dim)
    t = rand(dim)
    p = 1.0 / dot(s, t)
    st = s * t'
    ts = t * s'
    ss = s * s'
    Bs = bfgs_hessian(B, s, t)

    @test isposdef(B)
    @test all(s .> 0)
    @test all(t .> 0)
    @test issymmetric(Bs)
    @test isposdef(Bs)
    @test bfgs_hessian(B, s, t, 0.0) ≈ B
    @test bfgs_hessian(I, s, t) ≈ (I - p * (st .+ ts)) .+ (p^2 * dot(t, t) + p) .* ss
end



