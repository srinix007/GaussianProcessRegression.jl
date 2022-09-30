
const RT_PI_BY_2 = sqrt(pi) * 0.5

gauss_integ(a, b) = RT_PI_BY_2 * erf(a, b)
gauss_integ(xs, w, a, b) = (1 / w) * gauss_integ(w * (a - xs), w * (b - xs))
erf_integ(w, a, b) = 1 / (w^2) * (exp(-(w * (b - a))^2) - 1) +
                     2.0 * (RT_PI_BY_2 / w) * (b - a) * erf(w * (b - a))

function antideriv(::SquaredExp, xs, hp, a, b)
    integ = ones(eltype(xs), size(xs, 2))
    antideriv!(integ, SquaredExp(), xs, hp, a, b)
    return integ
end

function antideriv!(integ, ::SquaredExp, xs, hp, a, b)
    fill!(integ, one(eltype(integ)))
    ns = size(xs, 2)
    σ = hp[1]
    nl = size(xs, 1)
    ls = view(hp, 2:(nl+1))
    prefac = σ^2 * (RT_PI_BY_2^nl) * prod(1 ./ ls)
    @inbounds for j = 1:ns
        @inbounds @simd for i = 1:nl
            l = ls[i]
            xij = xs[i, j]
            integ[j] *= erf(l * (a[i] - xij), l * (b[i] - xij))
        end
    end
    integ .*= prefac
    return nothing
end

function antideriv2(::SquaredExp, hp, a, b)
    integ2 = one(eltype(hp))
    dim = length(a)
    ls = view(hp, 2:(dim+1))
    @inbounds @simd for i in eachindex(ls)
        @inbounds integ2 *= erf_integ(ls[i], a[i], b[i])
    end
    return integ2 * hp[1]^2
end

function antideriv2!(integ2, ::SquaredExp, hp, a, b)
    integ2[1] = antideriv2(SquaredExp(), hp, a, b)
    return nothing
end

## High level API

function integrate(md::AbstractGPRModel, a, b; sample_noise=nothing)
    integrate(md, md.params, a, b; sample_noise=sample_noise)
end

function integrate(md::AbstractGPRModel, hp, a, b; sample_noise=nothing)
    wc = WtCache(md)
    ac = AntiDerivCache(md)
    Iout = similar(md.y, size(md.y, 2))
    var_Iout = similar(Iout)
    integrate!(Iout, var_Iout, md, hp, a, b, sample_noise, wc, ac)
    return Iout, var_Iout
end

## Low level API

function update_cache!(wc::AbstractWtCache, md::AbstractGPRModel, hp, ::Nothing)
    kernel!(wc.kxx, md.covar, hp, md.x)
    kchol = cholesky!(Hermitian(wc.kxx))
    ldiv!(wc.wt, kchol, md.y)
    return nothing
end

function update_cache!(wc::AbstractWtCache, md::AbstractGPRModel, hp, sample_noise)
    kernel!(wc.kxx, md.covar, hp, md.x)
    # W = P(λ + σ)⁻¹P⁻¹y
    eig = Eigen(LAPACK.syevr!(wc.ws, 'V', 'A', 'U', wc.kxx, 0.0, 0.0, 0, 0, -1.0)...)
    wc.λ .= eig.values
    wc.P .= eig.vectors
    inverse_diagonal_update!(wc.wt, wc.λ, wc.P, sample_noise, md.y, wc.tmp)
    return nothing
end

function inverse_diagonal_update!(ABy, λ, P, ϵ::Float64, y, tmp)
    tmp .= 1.0 ./ (λ .+ ϵ)
    mul!(ABy, P', y)
    tmp .*= ABy
    mul!(ABy, P, tmp)
    return nothing
end

function inverse_diagonal_update!(ABy, λ, P, ϵ::AbstractVector, y, tmp)
    na = [CartesianIndex()]
    tmp .= 1.0 ./ (λ[:, na] .+ ϵ[na, :])
    mul!(ABy, P', y)
    tmp .*= ABy
    mul!(ABy, P, tmp)
    return nothing
end

function inverse_diagonal_update2!(λ, P, ϵ, y, tmp)
    mul!(tmp, P', y)
    tmp .*= tmp
    D = 1.0 ./ (λ .+ ϵ)
    return dot(tmp, D)
end

function inverse_diagonal_update2!(yABy, λ, P, ϵ::AbstractVector, y, tmp)
    na = [CartesianIndex()]
    mul!(tmp, P', y)
    tmp .*= tmp
    D = 1.0 ./ (λ[na, :] .+ ϵ[:, na])
    mul!(yABy, D, tmp)
    return nothing
end

function update_cache!(ac::AbstractAntiDerivCache, md::AbstractGPRModel, hp, a, b)
    antideriv!(ac.k1, SquaredExp(), md.x, hp, a, b)
    antideriv2!(ac.k2, SquaredExp(), hp, a, b)
    return nothing
end

function integrate!(Iout, var_Iout, md::AbstractGPRModel, hp, a, b, sample_noise, wc::AbstractWtCache,
    ac::AbstractAntiDerivCache)
    update_cache!(wc, md, hp, sample_noise)
    update_cache!(ac, md, hp, a, b)
    integrate!(Iout, var_Iout, sample_noise, wc, ac)
    return nothing
end

function integrate!(Iout, var_Iout, sample_noise, wc::AbstractWtCache, ac::AbstractAntiDerivCache)
    mean_integ_impl!(Iout, wc, ac)
    var_integ_impl!(var_Iout, sample_noise, wc, ac)
    return nothing
end

function mean_integ_impl!(Iout, wc::AbstractWtCache, ac::AbstractAntiDerivCache)
    mean_integ_impl!(Iout, wc.wt, ac.k1)
end

function mean_integ_impl!(Iout, wt, k1)
    mul!(Iout, wt', k1)
    return nothing
end


function var_integ_impl!(var_Iout, ::Nothing, wc::AbstractWtCache, ac::AbstractAntiDerivCache)
    kchol = Cholesky(UpperTriangular(wc.kxx))
    var_integ_impl!(var_Iout, nothing, kchol, ac.k1, ac.k2, wc.tmp)
    return nothing
end

function var_integ_impl!(var_Iout, ::Nothing, kchol, k1, k2, tmp)
    @views tt = tmp[:, 1]
    tt .= k1
    ldiv!(kchol.L, tt)
    var_Iout[1] = k2[1] - dot(tt, tt)
    return nothing
end

function var_integ_impl!(var_Iout, sample_noise, wc::AbstractWtCache, ac::AbstractAntiDerivCache)
    var_integ_impl!(var_Iout, sample_noise, wc.λ, wc.P, ac.k1, ac.k2, wc.tmp)
    return nothing
end

function var_integ_impl!(var_Iout, sample_noise, λ, P, k1, k2, tmp)
    @views tt = tmp[:, 1]
    tt .= k1
    inverse_diagonal_update2!(var_Iout, λ, P, sample_noise, k1, tt)
    var_Iout .= k2[1] .- var_Iout
    return nothing
end
