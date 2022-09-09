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