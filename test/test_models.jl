
@testset verbose = true "GPR Prediction" for n in (100, 200, 500),
                                             np in (100, 200, 500),
                                             dim in (1, 2, 5)

    x = rand(dim, n)
    xp = rand(dim, np)
    y = dropdims(sin.(sum(x; dims = 1)) .^ 2; dims = 1)
    yp = dropdims(sin.(sum(xp; dims = 1)) .^ 2; dims = 1)

    @testset "Type Stability" begin
        md = @inferred GPRModel(SquaredExp() + WhiteNoise(), x, y)
        @inferred predict_mean(md, xp)
        @inferred predict(md, xp)
    end

    @testset "Numerics" begin
        md2 = @inferred GPRModel(SquaredExp() + SquaredExp(), x, y)
        @test predict_mean(md2, x) ≈ y rtol = 1e-7

        μₚ = similar(y)
        Σₚ = similar(y, size(y, 1), size(y, 1))
        μₚ, Σₚ = predict(md2, x)
        @test Σₚ ≈ zeros(size(y, 1), size(y, 1)) atol = 1e-7

        md3 = @inferred GPRModel(SquaredExp() + WhiteNoise(), x, y)
        md3.params[end] = 1e-5
        @test predict_mean(md3, x) ≈ y rtol = 1e-3

        μₚ, Σₚ = predict(md3, x)
        @test Σₚ ≈ zeros(size(y, 1), size(y, 1)) atol = 1e-3
    end

    @testset "Covar diagonal" begin
        md = GPRModel(SquaredExp() + WhiteNoise(), x, y)
        Σd = similar(xp, size(xp, 2))
        yp, Σf = predict(md, xp)
        pc = GPRPredictCache(md, size(xp, 2))
        update_cache!(pc, md)
        predict!(yp, Diagonal(Σd), md, xp, pc)
        @test diag(Σf) ≈ Σd atol = 1e-5

        md2 = GPRModel(SquaredExp(), x, y)
        yp, Σf = predict(md2, xp)
        update_cache!(pc, md2)
        predict!(yp, Diagonal(Σd), md2, xp, pc)
        @test diag(Σf) ≈ Σd atol = 1e-5
    end
end
