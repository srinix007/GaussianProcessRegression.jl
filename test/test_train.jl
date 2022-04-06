
covars = (SquaredExp() + WhiteNoise(), SquaredExp())

@testset "Train" verbose = true begin
    @testset "Model Train $i $dim $n" verbose = true for i in eachindex(covars),
                                                         dim = 5:9,
                                                         n = 100:100:500

        cov = covars[i]
        x = rand(dim, n)
        gp = GaussianProcess(x -> zero(eltype(x)), cov)
        hp = rand(0.1:0.1:5.0, dim_hp(cov, dim))
        hp[1] = 1.0
        hp[end] = 1e-4
        y = sample(gp(x, hp))
        mdl = GPRModel(cov, x, y)
        hp0 = ones(length(hp))
        ϵJ = 1e-2
        options = Optim.Options(; g_tol = ϵJ, show_trace = false, iterations = 200)
        hp_tr, res = train(mdl, MarginalLikelihood(), hp0; method = NewtonTrustRegion(),
                           options = options)
        @test Optim.g_converged(res) broken = !Optim.g_converged(res)
    end
end
