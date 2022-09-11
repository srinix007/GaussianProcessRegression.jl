predict_cache(::GPRModel, ::Cmap) = GPRSplitPredictCache

alloc_kernel(cov::AbstractKernel, xp::Cmap, x) = SplitKernel(cov, x, xp)

function predict_mean_impl!(μₚ, Kxp::SplitKernel, wt, pc::GPRSplitPredictCache)
    predict_split_mean_impl!(μₚ, Kxp.A, Kxp.B, Kxp.C, wt, pc.Cw, pc.BCw)
    return nothing
end

function predict_split_mean_impl!(μₚ, A, B, C, wt, Cw, BCw)
    fill!(μₚ, zero(eltype(μₚ)))
    for k in axes(A, 3)
        @views mul!(Cw, Diagonal(wt), C[:, :, k])
        @views mul!(BCw, B[:, :, k], Cw)
        @views BCw .*= A[:, :, k]
        μₚ .+= BCw
    end
    return nothing
end

#=
function predict_covar_impl!(Σₚ::Diagonal, SKxp::SplitKernel, kchol, pc::GPRSplitPredictCache)
    sr = 1:size(SKxp, 3)
    er = 1:size(SKxp, 1)
    qr = 1:size(SKxp, 2)
    nq = length(qr)
    for e in er
        @inbounds for q in qr, s in sr
            @inbounds pc.Kxq[q, s] = SKxp[e, q, s]
        end
        rn = (1+(e-1)*nq):(e*nq)
        @inbounds @views Σq = Diagonal(Σₚ.diag[rn])
        predict_covar_impl!(Σq, pc.Kxq, kchol)
    end
    return nothing
end
=#

function predict_covar_impl!(Σₚ::Diagonal, SKxp::SplitKernel, kchol, pc::GPRSplitPredictCache; er=1:3)
    na = [CartesianIndex()]
    nr = 1:size(SKxp, 4)
    nq = size(SKxp, 2)
    for e in er
        fill!(pc.Kxq, zero(eltype(pc.Kxq)))
        for n in nr
            pc.Kxq .+= SKxp.A[e, :, na, n] .* SKxp.B[e, na, :, n] .* SKxp.C[:, :, n]'
        end
        rn = (1+(e-1)*nq):(e*nq)
        Σq = Diagonal(view(Σₚ.diag, rn))
        predict_covar_impl!(Σq, pc.Kxq, kchol, pc)
    end
    return nothing
end