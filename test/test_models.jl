
@testset verbose=true "GPR Prediction" for n in (100,200,500), np in (100,200,500), dim in (1,2,5)
    x = rand(dim,n)
    xp = rand(dim,np)
    y = dropdims(sin.(sum(x, dims=1)).^2, dims=1)
    yp = dropdims(sin.(sum(xp, dims=1)).^2, dims=1)


    @testset "Type Stability" begin
        md = @inferred GPRModel(SquaredExp() + WhiteNoise(), x, y)
        @inferred predict!(similar(yp), md, xp)
        @inferred predict(md, xp)
        @inferred update!(md)
    end

    @testset "Numerics" begin
        md2 = @inferred GPRModel(SquaredExp() + SquaredExp(), x, y)
        @test predict(md2, x) ≈ y rtol = 1e-7
        md3 = @inferred GPRModel(SquaredExp() + WhiteNoise(), x, y)
        md3.params[end] = 1e-5
        @test predict(md3, x) ≈ y rtol = 1e-3
    end
end