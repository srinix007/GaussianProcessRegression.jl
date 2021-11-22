covs = (SquaredExp(), SquaredExp() + WhiteNoise(),
        SquaredExp() + SquaredExp() + WhiteNoise())

@testset verbose = true "Split Kernel" for cov in covs,
                                           dim in (2, 3, 5),
                                           n in (100, 200, 26)

    x = rand(dim, n)
    xe = rand(dim, n + rand((100, -10)))
    xq = rand(dim, n + rand((100, -10)))
    xeq = Cmap(+, xe, xq)

    @testset "Cmap" begin
        for i in size(xe, 2), j in size(xq, 2)
            @test xeq[i, j] ≈ xe[:, i] .+ xq[:, j]
        end
        for j in size(xq, 2)
            @test xeq[:, j] ≈ reduce(hcat, [xe[:, i] .+ xq[:, j] for i = 1:size(xe, 2)])
        end
        for i in size(xe, 2)
            @test xeq[i, :] ≈ reduce(hcat, [xe[:, i] .+ xq[:, j] for j = 1:size(xq, 2)])
        end
        @test xeq[:, :] ≈ reduce(hcat,
                                 [xe[:, i] .+ xq[:, j] for j = 1:size(xq, 2) for i = 1:size(xe, 2)])
    end

    @testset "SplitDistance" begin
        DA = distance(SplitDistanceA(), xe, xq)
        DB = distance(Euclidean(), xe, x)
        DC = distance(SplitDistanceC(), x, xq)
        D = distance(Euclidean(), xeq[:, :], x)
        for e = 1:size(xe, 2), q = 1:size(xq, 2), s in size(x, 2)
            idx = LinearIndices((size(xe, 2), size(xq, 2)))[e, q]
            @test D[idx, s] ≈ DA[e, q] + DB[e, s] + DC[s, q]
        end
    end

end
