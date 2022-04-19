
gaussian(x, xs, w) = exp(-w^2 * (x - xs)^2)

@testset "Definite Gaussian integrals" begin
    for i = 1:100
        xs = 3.0 * rand()
        w = 3.0 * rand()
        a = -3.0 + 6 * rand()
        b = -3.0 + 6 * rand()
        @testset "Gaussian integral" begin
            integ_quad = quadgk(x -> gaussian(x, xs, w), a, b; rtol = 1e-5)[1]
            integ_adrv = gauss_integ(xs, w, a, b)
            @test integ_quad ≈ integ_adrv
        end
        @testset "Erf integral" begin
            integ2_quad = quadgk(x -> gauss_integ(x, w, a, b), a, b; rtol = 1e-5)[1]
            integ2_adrv = erf_integ(w, a, b)
            @test integ2_quad ≈ integ2_adrv
        end
    end
end

@testset "SquaredExp antiderivative" begin
    @testset "dim=$dim n=$n" for dim = 2:5, n = 100:100:500
        xs = rand(dim, n)
        hp = 5.0 .* rand(dim_hp(SquaredExp(), dim))
        a = -2 .+ 4 .* rand(dim)
        b = a .+ 2.0 .* rand(dim)
        na = [CartesianIndex()]
        w = view(hp, 2:(dim+1))
        integ = hp[1] .*
                prod(gauss_integ.(xs[:, :], w[:, na], a[:, na], b[:, na]); dims = 1)
        integ_loop = antideriv(SquaredExp(), xs, hp, a, b)
        @test dropdims(integ; dims = 1) ≈ integ_loop
    end
end
