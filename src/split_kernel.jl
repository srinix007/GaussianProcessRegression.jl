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

struct SplitKernel{C<:Cmap,T,N,X<:AbstractArray{T},M<:AbstractArray{T,N}} <:
       AbstractArray{T,N}
    xp::C
    x::X
    A::M
    B::M
    C::M
    function SplitKernel(c::Cmap, x::H, A::J, B::J,
                         C::J) where {H<:AbstractArray,J<:AbstractArray}
        new{typeof(c),eltype(A),ndims(A),H,J}(c, x, A, B, C)
    end
end

function SplitKernel(x, cm::Cmap, nkrn)
    s = size(x, 2)
    e = size(cm, 2)
    q = size(cm, 3)
    A = similar(x, e, q, nkrn)
    B = similar(x, e, s, nkrn)
    C = similar(x, s, q, nkrn)
    return SplitKernel(cm, x, A, B, C)
end

function SplitKernel(cov::ComposedKernel, x, cm::Cmap)
    nkrn = length([krn for krn in cov.kernels if krn isa SquaredExp])
    return SplitKernel(x, cm, nkrn)
end

SplitKernel(::SquaredExp, x, cm::Cmap) = SplitKernel(x, cm, 1)

Base.size(Kxp::SplitKernel) = (size(Kxp.A, 1), size(Kxp.A, 2), size(Kxp.x, 2),
                               size(Kxp.A, 3))
Base.size(Kxp::SplitKernel, i) = size(Kxp)[i]

function Base.getindex(Kxp::SplitKernel, ::Colon, ::Colon, ::Colon)
    sr = 1:size(Kxp, 3)
    er = 1:size(Kxp, 1)
    qr = 1:size(Kxp, 2)
    nr = 1:size(Kxp, 4)
    KK = zeros(eltype(Kxp.A), length(er), length(qr), length(sr))
    @turbo for n in nr, q in qr, e in er, s in sr
        KK[e, q, s] += Kxp.A[e, q, n] * Kxp.B[e, s, n] * Kxp.C[s, q, n]
    end
    return KK
end

function Base.getindex(Kxp::SplitKernel, e::Int, q::Int, ::Colon)
    sr = 1:size(Kxp, 3)
    nr = 1:size(Kxp, 4)
    KK = zeros(eltype(Kxp.A), length(sr))
    @turbo for n in nr, s in sr
        KK[s] += Kxp.A[e, q, n] * Kxp.B[e, s, n] * Kxp.C[s, q, n]
    end
    return KK
end

function Base.getindex(Kxp::SplitKernel, e::Int, q::Int, s::Int)
    nr = 1:size(Kxp, 4)
    kk = zero(eltype(Kxp.A))
    @inbounds for n in nr
        kk += Kxp.A[e, q, n] * Kxp.B[e, s, n] * Kxp.C[s, q, n]
    end
    return kk
end

Base.show(io::IO, ::MIME"text/plain", Kxp::SplitKernel) = show(typeof(Kxp))
Base.show(io::IO, ::MIME"text/plain", Kxps::Vector{<:SplitKernel}) = println.(typeof.(Kxps))


struct SplitDistanceA <: AbstractDistanceMetric end
struct SplitDistanceC <: AbstractDistanceMetric end

function distance!(::SplitDistanceA, D, xe, xq)
    n = [CartesianIndex()]
    fill!(D, zero(eltype(D)))
    sum!(D, (xq[:, n, :] .^ 2 .+ 2.0 .* xe[:, :, n] .* xq[:, n, :]), 1)
    return nothing
end

function distance!(::SplitDistanceC, D, xs, xq)
    n = [CartesianIndex()]
    fill!(D, zero(eltype(D)))
    sum!(D, (-2.0 .* xs[:, :, n] .* xq[:, n, :]), 1)
    return nothing
end

function kernel(cov::ComposedKernel, hp, xp::Cmap, x)
    Kxps = SplitKernel(cov, x, xp)
    kernel!(Kxps, cov, hp, xp, x)
    return Kxps
end

function kernel(::SquaredExp, hp, xp::Cmap, x)
    Kxps = SplitKernel(SquaredExp(), x, xp)
    kernel!(Kxps, SquaredExp(), hp, xp, x)
    return Kxps
end

function kernel!(Kxps::SplitKernel, cov::ComposedKernel, hp, xp::Cmap, x)
    hps = split(hp, [dim_hp(t, size(x, 1)) for t in cov.kernels])
    Ks, hpn = rm_noise(cov, hps)
    for i in eachindex(Ks)
        kernel!(Kxps, i, Ks[i], hpn[i], xp, x)
    end
    return nothing
end

function kernel!(Kxps::SplitKernel, ::SquaredExp, hp, xp::Cmap, x)
    kernel!(Kxps, 1, SquaredExp(), hp, xp, x)
    return nothing
end

function kernel!(Kxp::SplitKernel, nkrn::Int, ::SquaredExp, hp, xp::Cmap, x)
    hps = copy(hp)
    hps[1] = 1.0
    @views kernel!(Kxp.A[:, :, nkrn], SquaredExp(), hps, xp.xe, xp.xq;
                   dist = SplitDistanceA())
    @views kernel!(Kxp.B[:, :, nkrn], SquaredExp(), hps, xp.xe, x; dist = Euclidean())
    @views kernel!(Kxp.C[:, :, nkrn], SquaredExp(), hp, x, xp.xq; dist = SplitDistanceC())
    return nothing
end

function predict_split_mean!(μₚ, A, B, C, wt)
    Cw = similar(C)
    BCw = similar(A)
    predict_split_mean_impl!(μₚ, Cw, BCw, A, B, C, wt)
    return nothing
end

function predict_split_mean_impl!(μₚ, Cw, BCw, A, B, C, wt)
    μₚ .= A
    mul!(Cw, Diagonal(wt), C)
    mul!(BCw, B, Cw)
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
