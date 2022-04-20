
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

integrate(md::AbstractGPRModel, a, b) = integrate(md, md.params, a, b)

function integrate(md::AbstractGPRModel, hp, a, b)
    wc = WtCache(md)
    ac = AntiDerivCache(md)
    return integrate!(md, hp, a, b, wc, ac)
end

## Low level API

function update_cache!(wc::AbstractWtCache, md::AbstractGPRModel, hp)
    kernel!(wc.kxx, md.covar, hp, md.x)
    kchol = cholesky!(Hermitian(wc.kxx))
    ldiv!(wc.wt, kchol, md.y)
    return nothing
end

function update_cache!(ac::AbstractAntiDerivCache, md::AbstractGPRModel, hp, a, b)
    antideriv!(ac.k1, SquaredExp(), md.x, hp, a, b)
    antideriv2!(ac.k2, SquaredExp(), hp, a, b)
    return nothing
end

function integrate!(md::AbstractGPRModel, hp, a, b, wc::AbstractWtCache,
                    ac::AbstractAntiDerivCache)
    update_cache!(wc, md, hp)
    update_cache!(ac, md, hp, a, b)
    return integrate!(wc, ac)
end

function integrate!(wc::AbstractWtCache, ac::AbstractAntiDerivCache)
    μ_integ = mean_integ_impl(wc.wt, ac.k1)
    kchol = Cholesky(UpperTriangular(wc.kxx))
    σ2 = var_integ_impl!(ac.k1, ac.k2, kchol, wc.tmp)
    return μ_integ, sqrt(σ2)
end

function mean_integ_impl(wt, k1)
    return dot(wt, k1)
end

function var_integ_impl!(k1, k2, kchol, tmp)
    tmp .= k1
    ldiv!(kchol.L, tmp)
    return k2[1] - dot(tmp, tmp)
end
