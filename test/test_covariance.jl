kerns = (SquaredExp(),)

@testset "Covariance: $kern, $n, dim=$dim" for kern in kerns, n in (100, 200, 300), dim in 1:7 
    x = rand(dim, n)
    xp = rand(dim, 2 * n)
    hp = rand(dim_hp(kern, dim))

    Kxx = kernel(kern, hp, x)
    Kxp = kernel(kern, hp, x, xp)

    @testset "Structure" begin
        @test issymmetric(Kxx)
        @test isposdef(Kxx)
        @test size(Kxx) == (n, n)
        @test size(Kxp) == (n, 2 * n)
    end

    @testset "Compose" begin
        krnt = SquaredExp() + WhiteNoise()
        hp = rand(dim_hp(krnt, dim))
        @test kernel(krnt, hp, x) ≈ kernel(SquaredExp(), hp[1:end - 1], x) + hp[end]^2 * I
        @test kernel(krnt, hp, x, xp) ≈ kernel(SquaredExp(), hp[1:end - 1], x, xp)

        krnt = SquaredExp() + SquaredExp()
        hp = rand(dim_hp(krnt, dim))
        @test kernel(krnt, hp, x) ≈ kernel(SquaredExp(), hp[1:(dim + 1)], x) .+ kernel(SquaredExp(), hp[(dim + 2):end], x)
        @test kernel(krnt, hp, x, xp) ≈ kernel(SquaredExp(), hp[1:(dim + 1)], x, xp) .+ kernel(SquaredExp(), hp[(dim + 2):end], x, xp)

        krnt = SquaredExp() + SquaredExp() + WhiteNoise()
        hp = rand(dim_hp(krnt, dim))
        @test kernel(krnt, hp, x) ≈ kernel(SquaredExp(), hp[1:(dim + 1)], x) .+ kernel(SquaredExp(), hp[(dim + 2):end - 1], x) + hp[end]^2 * I
        @test kernel(krnt, hp, x, xp) ≈ kernel(SquaredExp(), hp[1:(dim + 1)], x, xp) .+ kernel(SquaredExp(), hp[(dim + 2):end - 1], x, xp)

        krnt = WhiteNoise() + SquaredExp()
        hp = rand(dim_hp(krnt, dim))
        @test kernel(krnt, hp, x) ≈ kernel(SquaredExp(), hp[1:end - 1], x) + hp[end]^2 * I
        @test kernel(krnt, hp, x, xp) ≈ kernel(SquaredExp(), hp[1:end - 1], x, xp)

        krnt = SquaredExp() + WhiteNoise() + SquaredExp()
        hp = rand(dim_hp(krnt, dim))
        @test kernel(krnt, hp, x) ≈ kernel(SquaredExp(), hp[1:(dim + 1)], x) .+ kernel(SquaredExp(), hp[(dim + 2):end - 1], x) + hp[end]^2 * I
        @test kernel(krnt, hp, x, xp) ≈ kernel(SquaredExp(), hp[1:(dim + 1)], x, xp) .+ kernel(SquaredExp(), hp[(dim + 2):end - 1], x, xp)

    end


end







        