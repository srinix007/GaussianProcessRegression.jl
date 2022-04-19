
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
    ns = size(xs, 2)
    σ = hp[1]
    nl = size(xs, 1)
    ls = view(hp, 2:(nl+1))
    prefac = σ * (RT_PI_BY_2^nl) * prod(1 ./ ls)
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
    return integ2 * hp[1]
end
