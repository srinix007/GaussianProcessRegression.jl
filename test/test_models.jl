
@testset verbose = true "GPR Prediction" for n in (100, 200, 500),
                                             np in (100, 200, 500),
                                             dim in (1, 2, 5)

    x = rand(dim, n)
    xp = rand(dim, np)
    y = dropdims(sin.(sum(x; dims=1)) .^ 2; dims=1)
    yp = dropdims(sin.(sum(xp; dims=1)) .^ 2; dims=1)

    @testset "Type Stability" begin
        md = @inferred GPRModel(SquaredExp() + WhiteNoise(), x, y)
        @inferred predict!(similar(yp), md, xp)
        @inferred predict(md, xp)
        @inferred update_params!(md, rand(size(md.params)...))
    end

    @testset "update_params" begin
        md1 = GPRModel(SquaredExp() + WhiteNoise(), x, y)
        kxx = kernel(md1.covar, md1.params, x)
        kxx_chol = cholesky(kxx)
        @test md1.cache.kxx_chol.U ≈ kxx_chol.U
        @test md1.cache.kxx_chol.L ≈ kxx_chol.L
        hp_new = md1.params .+ 1.0
        update_params!(md1, hp_new)
        @test md1.params ≈ hp_new
        kxx = kernel(md1.covar, hp_new, x)
        kxx_chol = cholesky(kxx)
        @test md1.cache.kxx_chol.U ≈ kxx_chol.U
        @test md1.cache.kxx_chol.L ≈ kxx_chol.L
    end

    @testset "Numerics" begin
        md2 = @inferred GPRModel(SquaredExp() + SquaredExp(), x, y)
        @test predict(md2, x) ≈ y rtol = 1e-7

        μₚ = similar(y)
        Σₚ = similar(y, size(y, 1), size(y, 1))
        predict!(μₚ, Σₚ, md2, x)
        @test Σₚ ≈ zeros(size(y, 1), size(y, 1)) atol = 1e-7

        md3 = @inferred GPRModel(SquaredExp() + WhiteNoise(), x, y)
        md3.params[end] = 1e-5
        GaussianProcessRegression.update_cache!(md3.cache, md3)
        @test predict(md3, x) ≈ y rtol = 1e-3

        predict!(μₚ, Σₚ, md3, x)
        @test Σₚ ≈ zeros(size(y, 1), size(y, 1)) atol = 1e-3
    end
end