
@testset verbose = true "BFGS Quad dim=$dim" for dim = 2:30
    J = rand(dim)
    L = LowerTriangular(rand(dim, dim))
    H = Hermitian(L * L' + 1e-7 * I)
    x0 = rand(dim)
    jac = let J = J, H = H
        x -> J + H * x
    end
    xma = -H \ J
    eps = 1e-4
    max_iter = 100000

    @testset "Hessian finite diff" begin
        @test isposdef(H)
        @test hessian_fd(x -> J + H * x, rand(dim)) ≈ H atol = eps
    end

    @testset "Quadratic optimization" begin
        xm, Jm, Hm, iters = bfgs_quad(x0, jac(x0), I, jac; ϵ = eps, max_iter = max_iter)
        @test iters < max_iter
        @test xm ≈ xma rtol = 10 * eps
        @test norm(Jm) < eps
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

covars = (SquaredExp() + WhiteNoise(),)

@testset "Model Update $dim $n $i" for dim = 3:6, n = 100:100:500, cov in covars, i = 1:2
    x = rand(dim, n)
    gp = GaussianProcess(x -> zero(eltype(x)), cov)
    hp = rand(0.1:0.1:5.0, dim_hp(cov, dim))
    hp[1] = 1.0
    y = sample(gp(x, hp))
    mdl = GPRModel(cov, x, y)
    hp0 = ones(length(hp))
    ϵJ = 1e-3
    options = Optim.Options(; g_tol = ϵJ, show_trace = false)
    hp_tr, res = train(mdl, MarginalLikelihood(), hp0; method = NewtonTrustRegion(),
                       options = options)
    converged = Optim.g_converged(res)
    mdl.params .= hp_tr
    δy = 0.01 .* y .^ 2
    if converged
        iters = update_sample!(mdl, δy, BFGSQuad(), MarginalLikelihood(), ϵJ)
        J = mdl.params .* grad(MarginalLikelihood(), mdl.params, mdl)
    end
    @test norm(J) < ϵJ skip = !converged
    @test iters < 10 skip = !converged
    #println("iters:", iters)
end


