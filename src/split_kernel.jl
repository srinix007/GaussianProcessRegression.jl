struct Cmap{F<:Function,T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    op::F
    xe::A
    xq::A
    function Cmap(op, xe::B, xq::B) where {B<:AbstractArray}
        new{typeof(op),eltype(xe),ndims(xe),B}(op, xe, xq)
    end
end

Base.size(c::Cmap) = (size(c.xe, 1), size(c.xe, 2), size(c.xq, 2))
Base.size(c::Cmap, i) = size(c)[i]
function Base.getindex(c::Cmap, i, j)
    na = [CartesianIndex()]
    reshape(c.op.(c.xe[:, i, na], c.xq[:, na, j]), size(c.xe, 1), :)
end

Base.show(io::IO, ::MIME"text/plain", cm::Cmap) = show(typeof(cm))

struct SplitKernel{C<:Cmap,T,N,M<:AbstractArray{T,N}} <: AbstractArray{T,N}
    xp::C
    x::M
    A::M
    B::M
    C::M
    function SplitKernel(c::Cmap, x::X, A::X, B::X, C::X) where {X<:AbstractArray}
        new{typeof(c),eltype(A),ndims(A),X}(c, x, A, B, C)
    end
end

Base.size(Kxp::SplitKernel) = (size(Kxp.x, 2), size(Kxp.A)...)
Base.size(Kxp::SplitKernel, i) = size(Kxp)[i]

function Base.getindex(Kxp::SplitKernel, s, e, q)
    sr = s isa Colon ? (1:size(Kxp, 1)) : (1:1)
    er = e isa Colon ? (1:size(Kxp, 2)) : (1:1)
    qr = q isa Colon ? (1:size(Kxp, 3)) : (1:1)
    KK = similar(Kxp.A, length(sr), length(er), length(qr))
    @turbo for q in qr, e in er, s in sr
        KK[s, e, q] = Kxp.A[e, q] * Kxp.B[e, s] * Kxp.C[s, q]
    end
    return KK
end

Base.show(io::IO, ::MIME"text/plain", Kxp::SplitKernel) = show(typeof(Kxp))

function predict_split_mean!(μₚ, A, B, C, wt)
    μₚ .= A
    Cw = Diagonal(wt) * C
    BCw = B * Cw
    μₚ .*= BCw
    return nothing
end

function predict_split_covar!(Σₚ, md::AbstractGPRModel, Kxp::SplitKernel)
    nx = size(C, 1)
    kxp = similar(A, nx)

    for e in axes(A, 1), q in axes(A, 2)
        @views kxp .= A[e, q] .* B[e, :] .* C[:, q]
        K⁻¹kxp = kxp / md.cache.kxx_chol.U
        tt = dot(kxp, Kkxp)
        Σₚ[e, q] = kernel(md.cov, md.params, Kxp.xp[e, q]) - tt
    end

    return nothing
end
