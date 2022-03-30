
covars = (SquaredExp() + WhiteNoise(), SquaredExp())

@testset "Model Train" verbose = true begin
    @testset "$i $dim $n" for i in eachindex(covars), dim = 2:7, n = 100:100:500
        cov = covars[i]
        x = rand(dim, n)
        gp = GaussianProcess(x -> zero(eltype(x)), cov)
        hp = rand(0.1:0.1:5.0, dim_hp(cov, dim))
        hp[1] = 1.0
        y = sample(gp(x, hp))
        mdl = GPRModel(cov, x, y)
        hp0 = ones(length(hp))
        ϵJ = 1e-3
        options = Optim.Options(; g_tol = ϵJ, show_trace = true, iterations = 200)
        hp_tr, res = train(mdl, MarginalLikelihood(), hp0; method = NewtonTrustRegion(),
                           options = options)
        @test Optim.g_converged(res) broken = !Optim.g_converged(res)
        println(hp)
    end
end
