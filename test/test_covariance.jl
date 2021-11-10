kerns = (SquaredExp(),)

function grad_fd(kern, i, hp, x, ϵ=1e-7)
    K = kernel(kern, hp, x)
    hpϵ = copy(hp)
    hpϵ[i] += ϵ
    Kϵ = kernel(kern, hpϵ, x)
    return (Kϵ .- K) ./ ϵ
end

@testset verbose = true "Covariance: $kern, $n, dim=$dim" for kern in kerns,
                                                              n in (100, 200, 300),
                                                              dim in 1:7

    x = rand(dim, n)
    xp = rand(dim, 2 * n)
    hp = rand(dim_hp(kern, dim))

    Kxx = kernel(kern, hp, x)
    Kxp = kernel(kern, hp, x, xp)

    @testset "Type Stability" begin
        @inferred kernel(kern, hp, x)
        @inferred kernel(kern, hp, x, xp)
    end

    @testset "Structure" begin
        @test issymmetric(Kxx)
        @test isposdef(Kxx)
        @test size(Kxx) == (n, n)
        @test size(Kxp) == (n, 2 * n)
    end

    @testset "Compose" begin
        krnt = SquaredExp() + WhiteNoise()
        hps = rand(dim_hp(krnt, dim))
        @inferred kernel(krnt, hps, x)
        @inferred kernel(krnt, hps, x, xp)
        @test kernel(krnt, hps, x) ≈
              kernel(SquaredExp(), hps[1:(end - 1)], x) + hps[end]^2 * I
        @test kernel(krnt, hps, x, xp) ≈ kernel(SquaredExp(), hps[1:(end - 1)], x, xp)

        krnt = SquaredExp() + SquaredExp()
        hps = rand(dim_hp(krnt, dim))
        @inferred kernel(krnt, hps, x)
        @inferred kernel(krnt, hps, x, xp)
        @test kernel(krnt, hps, x) ≈
              kernel(SquaredExp(), hps[1:(dim + 1)], x) .+
              kernel(SquaredExp(), hps[(dim + 2):end], x)
        @test kernel(krnt, hps, x, xp) ≈
              kernel(SquaredExp(), hps[1:(dim + 1)], x, xp) .+
              kernel(SquaredExp(), hps[(dim + 2):end], x, xp)

        krnt = SquaredExp() + SquaredExp() + WhiteNoise()
        hps = rand(dim_hp(krnt, dim))
        @inferred kernel(krnt, hps, x)
        @inferred kernel(krnt, hps, x, xp)
        @test kernel(krnt, hps, x) ≈
              kernel(SquaredExp(), hps[1:(dim + 1)], x) .+
              kernel(SquaredExp(), hps[(dim + 2):(end - 1)], x) + hps[end]^2 * I
        @test kernel(krnt, hps, x, xp) ≈
              kernel(SquaredExp(), hps[1:(dim + 1)], x, xp) .+
              kernel(SquaredExp(), hps[(dim + 2):(end - 1)], x, xp)

        krnt = WhiteNoise() + SquaredExp()
        hps = rand(dim_hp(krnt, dim))
        @inferred kernel(krnt, hps, x)
        @inferred kernel(krnt, hps, x, xp)
        @test kernel(krnt, hps, x) ≈ kernel(SquaredExp(), hps[2:end], x) + hps[1]^2 * I
        @test kernel(krnt, hps, x, xp) ≈ kernel(SquaredExp(), hps[2:end], x, xp)

        krnt = SquaredExp() + WhiteNoise() + SquaredExp()
        hps = rand(dim_hp(krnt, dim))
        @inferred kernel(krnt, hps, x)
        @inferred kernel(krnt, hps, x, xp)
        @test kernel(krnt, hps, x) ≈
              kernel(SquaredExp(), hps[1:(dim + 1)], x) .+
              kernel(SquaredExp(), hps[(dim + 3):end], x) + hps[dim + 2]^2 * I
        @test kernel(krnt, hps, x, xp) ≈
              kernel(SquaredExp(), hps[1:(dim + 1)], x, xp) .+
              kernel(SquaredExp(), hps[(dim + 3):end], x, xp)
    end

    @testset "grad dim $i ($dim)" for i in dim
        @test grad(kern, i, hp, x) ≈ grad_fd(kern, i, hp, x) atol = 1e-3
        @inferred grad(kern, i, hp, x)
    end

    @testset "grad compose" begin
        dim = 2
        x = rand(dim, 100)
        krnt = SquaredExp() + WhiteNoise() + SquaredExp()
        hp = rand(dim_hp(krnt, dim))
        hps = split(hp, [dim + 1, 1, dim + 1])

        @test grad(krnt, 1, hp, x) ≈ grad(SquaredExp(), 1, hps[1], x)
        @test grad(krnt, 2, hp, x) ≈ grad(SquaredExp(), 2, hps[1], x)
        @test grad(krnt, 3, hp, x) ≈ grad(SquaredExp(), 3, hps[1], x)

        @test grad(krnt, 4, hp, x) ≈ grad(WhiteNoise(), 1, hps[2], x)

        @test grad(krnt, 5, hp, x) ≈ grad(SquaredExp(), 1, hps[3], x)
        @test grad(krnt, 6, hp, x) ≈ grad(SquaredExp(), 2, hps[3], x)
        @test grad(krnt, 7, hp, x) ≈ grad(SquaredExp(), 3, hps[3], x)
    end
end
