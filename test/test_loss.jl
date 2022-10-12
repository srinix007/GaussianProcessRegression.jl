@testset verbose = true "MarginalLikelihood" for n in (10, 20, 100)
    x = rand(n)
    y = rand(n)
    K = Diagonal(x)

    MLE = 0.5 * (dot(y, (1 ./ x) .* y) + sum(log.(x)) + n * log(2π))

    kchol = cholesky(collect(K))
    wt = kchol \ y
    @test loss(MarginalLikelihood(), kchol, y, wt) ≈ MLE
end

function grad_fd(ll, i, md, ϵ=1e-6)
    hp = md.params
    L = loss(ll, hp, md)
    hpϵ = copy(hp)
    hpϵ[i] += ϵ
    Lϵ = loss(ll, hpϵ, md)
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

    @test grad(ll, kchol, K, y, inv(K), copy(y)) ≈ -0.5 * (tr((y * y') * K) - size(K, 1)) rtol = 1e-5

    md = GPRModel(SquaredExp() + WhiteNoise(), x, y)
    @test loss(ll, md.covar, md.params, x, y) ≈ loss(ll, md.params, md)

    DL1 = grad(ll, md.params, md)

    tc = MllGradCache(md)
    update_cache!(tc, md.params, md)
    K = kernel(md.covar, md.params, md.x)
    kchol = cholesky(K)
    wt = kchol \ md.y
    Kin = inv(K)
    tt = similar(wt)
    @test kchol.U.data ≈ tc.kchol_base
    @test wt ≈ tc.α
    @test Kin ≈ tc.K⁻¹

    for i in eachindex(md.params)
        DK = grad(md.covar, i, md.params, md.x)
        DL = grad(ll, kchol, DK, wt, Kin, tt)
        @test DL ≈ grad_fd(ll, i, md) rtol = 1e-3
        @test DL1[i] ≈ grad_fd(ll, i, md) rtol = 1e-3
    end
end

@testset verbose = true "Grad MarginalLikelihood 2d y" for n in (10, 20, 100), dim in (2, 5), ne in (5, 10)
    x = rand(dim, n)
    y = similar(x, n, ne)
    for i in 1:ne
        y[:, i] = rand() .* dropdims(sum(sin.(x); dims=1); dims=1)
    end
    train_axis = rand(1:ne)
    hp = rand(dim_hp(SquaredExp(), dim))

    ll = MarginalLikelihood()
    cov = SquaredExp()
    K = kernel(cov, hp, x)
    kchol = cholesky(K)

    #@test grad(ll, kchol, K, y, inv(K), copy(y[:, train_axis])) ≈ -0.5 * (tr((y * y') * K) - size(K, 1)) rtol = 1e-5

    md = GPRModel(SquaredExp() + WhiteNoise(), x, y; train_axis=train_axis)
    @test loss(ll, md.covar, md.params, x, y[:, train_axis]) ≈ loss(ll, md.params, md)

    DL1 = grad(ll, md.params, md)

    tc = MllGradCache(md)
    update_cache!(tc, md.params, md)
    K = kernel(md.covar, md.params, md.x)
    kchol = cholesky(K)
    yt = get_sample(md)
    wt = kchol \ yt
    Kin = inv(K)
    tt = similar(wt)
    @test kchol.U.data ≈ tc.kchol_base
    @test wt ≈ tc.α
    @test Kin ≈ tc.K⁻¹

    for i in eachindex(md.params)
        DK = grad(md.covar, i, md.params, md.x)
        DL = grad(ll, kchol, DK, wt, Kin, tt)
        @test DL ≈ grad_fd(ll, i, md) rtol = 1e-3
        @test DL1[i] ≈ grad_fd(ll, i, md) rtol = 1e-3
    end
end
