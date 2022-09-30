
gaussian(x, xs, w) = exp(-w^2 * (x - xs)^2)

@testset "Definite Gaussian integrals" begin
    for i = 1:100
        xs = 3.0 * rand()
        w = 3.0 * rand()
        a = -3.0 + 6 * rand()
        b = -3.0 + 6 * rand()
        @testset "Gaussian integral" begin
            integ_quad = quadgk(x -> gaussian(x, xs, w), a, b; rtol=1e-5)[1]
            integ_adrv = gauss_integ(xs, w, a, b)
            @test integ_quad ≈ integ_adrv
        end
        @testset "Erf integral" begin
            integ2_quad = quadgk(x -> gauss_integ(x, w, a, b), a, b; rtol=1e-5)[1]
            integ2_adrv = erf_integ(w, a, b)
            @test integ2_quad ≈ integ2_adrv atol = 1e-5
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
        integ = hp[1]^2 .*
                prod(gauss_integ.(xs[:, :], w[:, na], a[:, na], b[:, na]); dims=1)
        integ_loop = antideriv(SquaredExp(), xs, hp, a, b)
        @test dropdims(integ; dims=1) ≈ integ_loop
    end
end

function gauss_leg_integ_3d(f, xg, wg)
    integ = zero(eltype(xg))
    for i in eachindex(xg), j in eachindex(xg), k in eachindex(xg)
        integ += wg[i] * wg[j] * wg[k] * f(xg[i], xg[j], xg[k])
    end
    return integ
end

@testset "Inverse diagonal update" begin
    for n in (100, 200, 300)
        L = LowerTriangular(rand(n, n))
        A = Hermitian(L * L' + 1e-7 * I)
        ϵ = 1e-5
        B = Diagonal(ϵ .* ones(n))
        y = rand(n)
        ABy = similar(y)
        ABy_exact = inv(A + B) * y
        tmp = similar(y)
        λ, P = eigen(A)
        GaussianProcessRegression.inverse_diagonal_update!(ABy, λ, P, ϵ, y, tmp)
        @test ABy ≈ ABy_exact rtol = 1e-6
    end
end

@testset "Inverse diagonal update vector noise" begin
    for n in (100, 200, 300), ne in (100, 200, 300)
        L = LowerTriangular(rand(n, n))
        A = Hermitian(L * L' + 1e-7 * I)
        ϵ = 1e-5 .* rand(ne)
        y = rand(n, ne)
        ABy = similar(y)
        ABy_exact = similar(y)
        tmp = similar(y)
        for i in 1:ne
            B = Diagonal(ϵ[i] .* ones(n))
            ABy_exact[:, i] .= inv(A + B) * y[:, i]
        end
        λ, P = eigen(A)
        GaussianProcessRegression.inverse_diagonal_update!(ABy, λ, P, ϵ, y, tmp)
        @test ABy ≈ ABy_exact rtol = 1e-5
    end
end

@testset "Inverse diagonal update quadratic" begin
    for n in (100, 200, 300)
        L = LowerTriangular(rand(n, n))
        A = Hermitian(L * L' + 1e-7 * I)
        ϵ = 1e-5
        B = Diagonal(ϵ .* ones(n))
        y = rand(n)
        yABy_exact = y' * inv(A + B) * y
        tmp = similar(y)
        λ, P = eigen(A)
        yABy = GaussianProcessRegression.inverse_diagonal_update2!(λ, P, ϵ, y, tmp)
        @test yABy ≈ yABy_exact rtol = 1e-6
    end
end

@testset "Inverse diagonal update quadratic vector" begin
    for n in (100, 200, 300), ne in (100, 200, 300)
        L = LowerTriangular(rand(n, n))
        A = Hermitian(L * L' + 1e-7 * I)
        ϵ = 1e-5 .* rand(ne)
        y = rand(n)
        yABy = similar(ϵ)
        yABy_exact = similar(yABy)
        tmp = similar(y)
        for i in 1:ne
            B = Diagonal(ϵ[i] .* ones(n))
            yABy_exact[i] = y' * inv(A + B) * y
        end
        λ, P = eigen(A)
        GaussianProcessRegression.inverse_diagonal_update2!(yABy, λ, P, ϵ, y, tmp)
        @test yABy ≈ yABy_exact rtol = 1e-5
    end
end

@testset "Integration with sample noise" begin
    @testset "Zero test dim=$dim, n=$n, k=$k" for dim in 1:4, n in (100, 200, 300), k in (100, 200, 300)
        x = rand(dim, n)
        y = rand(n, k)
        model = GPRModel(SquaredExp(), x, y)
        a, b = (zeros(dim), ones(dim))
        μ, σ = integrate(model, a, b, sample_noise=nothing)
        zero_noise = zeros(k)
        μ0, σ0 = integrate(model, a, b, sample_noise=zero_noise)
        @test μ ≈ μ0 rtol = 1e-5
        @test σ[1] ≈ σ0[2] rtol = 1e-4
        @test σ0[2:end] ≈ zeros(k - 1) atol = 1e-7
    end

    @testset "Inverse perturbation dim=$dim, n=$n, ne=$ne" for dim in 1:4, n in (100, 200, 300), ne in (100, 200, 300)
        x = rand(dim, n)
        y = rand(n, ne)
        model = GPRModel(SquaredExp(), x, y)
        noise = 1e-5 .* rand(ne)
        kxx = kernel(SquaredExp(), model.params, x)
        a, b = zeros(dim), ones(dim)
        ac = AntiDerivCache(model)
        update_cache!(ac, model, model.params, a, b)
        μ, Σ = integrate(model, a, b, sample_noise=noise)
        μ_exact = similar(μ)
        Σ_exact = similar(Σ)
        tmp = similar(y)
        ret = similar(y, 1)
        for i in 1:ne
            kxx_noise = kxx + Diagonal(noise[i] .* ones(n))
            kchol = cholesky(kxx_noise)
            @views wt = (kchol \ model.y[:, i])
            μ_exact[i] = dot(wt, ac.k1)
            GaussianProcessRegression.var_integ_impl!(ret, nothing, kchol, ac.k1, ac.k2, tmp)
            Σ_exact[i] = ret[1]
        end
        @test μ ≈ μ_exact rtol = 1e-5
        @test Σ ≈ Σ_exact rtol = 1e-5
    end
end

@testset "GP Integration" begin
    @testset "dim = 3 n = $n" for n = 100:100:500
        x = rand(3, n)
        y = dropdims(sin.(prod(x; dims=1)) .^ 2; dims=1)
        mds = GPRModel(SquaredExp(), x, y)
        hpmin, res = train(mds, MarginalLikelihood(); method=NewtonTrustRegion(),
            options=Optim.Options(; g_tol=1e-3))
        mds.params .= hpmin
        μ, σ = integrate(mds, zeros(3), ones(3))
        xg, wg = gauss(20, 0, 1)
        gauss_integral = gauss_leg_integ_3d((x, y, z) -> sin(x * y * z)^2, xg, wg)
        @test μ[1] ≈ gauss_integral atol = 3sqrt(σ[1])
    end
end