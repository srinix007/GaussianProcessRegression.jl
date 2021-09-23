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

end







        