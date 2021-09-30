kerns = (SquaredExp(),)

function grad_fd(kern, i, hp, x, ϵ=1e-7)
    K = kernel(kern, hp, x)
    hpϵ = copy(hp) 
    hpϵ[i] += ϵ
    Kϵ = kernel(kern, hpϵ, x) 
    return (Kϵ .- K) ./ ϵ
end


@testset verbose = true "Covariance: $kern, $n, dim=$dim" for kern in kerns, n in (100, 200, 300), dim in 1:7 
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
        hps = rand(dim_hp(krnt, dim))
        @test kernel(krnt, hps, x) ≈ kernel(SquaredExp(), hps[1:end - 1], x) + hps[end]^2 * I
        @test kernel(krnt, hps, x, xp) ≈ kernel(SquaredExp(), hps[1:end - 1], x, xp)

        krnt = SquaredExp() + SquaredExp()
        hps = rand(dim_hp(krnt, dim))
        @test kernel(krnt, hps, x) ≈ kernel(SquaredExp(), hps[1:(dim + 1)], x) .+ kernel(SquaredExp(), hps[(dim + 2):end], x)
        @test kernel(krnt, hps, x, xp) ≈ kernel(SquaredExp(), hps[1:(dim + 1)], x, xp) .+ kernel(SquaredExp(), hps[(dim + 2):end], x, xp)

        krnt = SquaredExp() + SquaredExp() + WhiteNoise()
        hps = rand(dim_hp(krnt, dim))
        @test kernel(krnt, hps, x) ≈ kernel(SquaredExp(), hps[1:(dim + 1)], x) .+ kernel(SquaredExp(), hps[(dim + 2):end - 1], x) + hps[end]^2 * I
        @test kernel(krnt, hps, x, xp) ≈ kernel(SquaredExp(), hps[1:(dim + 1)], x, xp) .+ kernel(SquaredExp(), hps[(dim + 2):end - 1], x, xp)

        krnt = WhiteNoise() + SquaredExp()
        hps = rand(dim_hp(krnt, dim))
        @test kernel(krnt, hps, x) ≈ kernel(SquaredExp(), hps[1:end - 1], x) + hps[end]^2 * I
        @test kernel(krnt, hps, x, xp) ≈ kernel(SquaredExp(), hps[1:end - 1], x, xp)

        krnt = SquaredExp() + WhiteNoise() + SquaredExp()
        hps = rand(dim_hp(krnt, dim))
        @test kernel(krnt, hps, x) ≈ kernel(SquaredExp(), hps[1:(dim + 1)], x) .+ kernel(SquaredExp(), hps[(dim + 2):end - 1], x) + hps[end]^2 * I
        @test kernel(krnt, hps, x, xp) ≈ kernel(SquaredExp(), hps[1:(dim + 1)], x, xp) .+ kernel(SquaredExp(), hps[(dim + 2):end - 1], x, xp)

    end

    @testset "grad dim $i ($dim)" for i in dim
        @test grad(kern, i, hp, x) ≈ grad(kern, i, hp, x) atol = 1e-3
    end

end







        