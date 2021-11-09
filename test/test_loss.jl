@testset verbose = true "MarginalLikelihood" for n in (10, 20, 100)
    x = rand(n)
    y = rand(n)
    K = Diagonal(x)

    MLE = 0.5 * (dot(y, (1 ./ x) .* y) + sum(log.(x)) + n * log(2π))

    @test loss(MarginalLikelihood(), collect(K), y) ≈ MLE
end

function grad_fd(ll, cov, i, hp, x, y, ϵ=1e-5)
    L = loss(ll, cov, hp, x, y)
    hpϵ = copy(hp)
    hpϵ[i] += ϵ
    Lϵ = loss(ll, cov, hpϵ, x, y)
    return (Lϵ .- L) ./ ϵ
end

@testset verbose = true "Grad MarginalLikelihood" for n in (10, 20, 100), dim in (2, 5)
    x = rand(dim, n)
    y = dropdims(sum(sin.(x); dims=1); dims=1)
    hp = rand(dim_hp(SquaredExp(), dim))

    ll = MarginalLikelihood()
    cov = SquaredExp()
    K = kernel(cov, hp, x)
    kchol = cholesky(K)

    @test grad(ll, kchol, K, y) ≈ -0.5 * (tr((y * y') * K) - size(K, 1)) rtol = 1e-5
end
