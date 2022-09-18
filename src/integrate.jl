
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
    kchol = cholesky!(Hermitian(wc.kxx))
    # K⁻¹y + K⁻¹ Σ K⁻¹y = W + K⁻¹ Σ W = W + K⁻¹ (Σ ⊗ W)
    ldiv!(wc.wt, kchol, md.y)         # W = K⁻¹y
    wc.tmp .= sample_noise' .* wc.wt  # T = Σ ⊗ W
    ldiv!(kchol, wc.tmp)              # T = K⁻¹T
    wc.wt .+= wc.tmp                  # W = W + T 
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
    mean_integ_impl(Iout, wc.wt, ac.k1)
    kchol = Cholesky(UpperTriangular(wc.kxx))
    var_integ_impl!(var_Iout, sample_noise, ac.k1, ac.k2, kchol, wc.tmp)
    return nothing
end

function mean_integ_impl(Iout, wt, k1)
    mul!(Iout, wt', k1)
    return nothing
end

function var_integ_impl!(var_Iout, ::Nothing, k1, k2, kchol, tmp)
    @views tt = tmp[:, 1]
    tt .= k1
    ldiv!(kchol.L, tt)
    var_Iout[1] = k2[1] - dot(tt, tt)
    return nothing
end

function var_integ_impl!(var_Iout, sample_noise, k1, k2, kchol, tmp)
    @views tt = tmp[:, 1]
    var_integ_impl!(var_Iout, nothing, k1, k2, kchol, tt)
    t1 = var_Iout[1]
    tt .= k1
    ldiv!(kchol, tt)
    tt .*= tt
    mul!(var_Iout, sample_noise, tt)
    var_Iout .= t1 .- var_Iout
    return nothing
end
