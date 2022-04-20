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

struct SplitKernel{T,N,M<:AbstractArray{T,N}} <: AbstractArray{T,N}
    A::M
    B::M
    C::M
    function SplitKernel(A::J, B::J, C::J) where {J<:AbstractArray}
        new{eltype(A),ndims(A),J}(A, B, C)
    end
end

function SplitKernel(x, ns, ne, nq, nkrn)
    A = similar(x, ne, nq, nkrn)
    B = similar(x, ne, ns, nkrn)
    C = similar(x, ns, nq, nkrn)
    return SplitKernel(A, B, C)
end

function SplitKernel(x, cm::Cmap, nkrn)
    s = size(x, 2)
    e = size(cm, 2)
    q = size(cm, 3)
    SplitKernel(x, s, e, q, nkrn)
end

function SplitKernel(cov::ComposedKernel, x, ne, nq)
    nkrn = length([krn for krn in cov.kernels if krn isa SquaredExp])
    ns = size(x, 2)
    return SplitKernel(x, ns, ne, nq, nkrn)
end

function SplitKernel(cov::ComposedKernel, x, cm::Cmap)
    ne = size(cm, 2)
    nq = size(cm, 3)
    SplitKernel(cov, x, ne, nq)
end

SplitKernel(::SquaredExp, x, ne, nq) = SplitKernel(x, size(x, 2), ne, nq, 1)

SplitKernel(::SquaredExp, x, cm::Cmap) = SplitKernel(x, cm, 1)

Base.size(Kxp::SplitKernel) = (size(Kxp.A, 1), size(Kxp.A, 2), size(Kxp.C, 1),
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

predict_cache(::GPRModel, ::Cmap) = GPRSplitPredictCache

alloc_kernel(cov::AbstractKernel, xp::Cmap, x) = SplitKernel(cov, x, xp)

predict_mean_impl!(μₚ, Kxp::SplitKernel, wt) = predict_split_mean!(μₚ, Kxp.A, Kxp.B, Kxp.C,
                                                                   wt)

function predict_split_mean!(μₚ, A, B, C, wt)
    Cw = similar(C, size(C)[1:end-1]...)
    BCw = similar(A, size(A)[1:end-1]...)
    predict_split_mean_impl!(μₚ, Cw, BCw, A, B, C, wt)
    return nothing
end

function predict_split_mean_impl!(μₚ, Cw, BCw, A, B, C, wt)
    fill!(μₚ, zero(eltype(μₚ)))
    for k in axes(A, 3)
        @views mul!(Cw, Diagonal(wt), C[:, :, k])
        @views mul!(BCw, B[:, :, k], Cw)
        @views BCw .*= A[:, :, k]
        μₚ .+= BCw
    end
    return nothing
end