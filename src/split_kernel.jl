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

function Base.getindex(Kxp::SplitKernel, ::Colon, ::Colon, ::Colon)
    sr = 1:size(Kxp, 1)
    er = 1:size(Kxp, 2)
    qr = 1:size(Kxp, 3)
    KK = similar(Kxp.A, length(sr), length(er), length(qr))
    @turbo for q in qr, e in er, s in sr
        KK[s, e, q] = Kxp.A[e, q] * Kxp.B[e, s] * Kxp.C[s, q]
    end
    return KK
end

function Base.getindex(Kxp::SplitKernel, ::Colon, e::Int, q::Int)
    sr = 1:size(Kxp, 1)
    KK = similar(Kxp.A, length(sr))
    @turbo for s in sr
        KK[s] = Kxp.A[e, q] * Kxp.B[e, s] * Kxp.C[s, q]
    end
    return KK
end

function Base.getindex(Kxp::SplitKernel, s::Int, e::Int, q::Int)
    return Kxp.A[e, q] * Kxp.B[e, s] * Kxp.C[s, q]
end

Base.show(io::IO, ::MIME"text/plain", Kxp::SplitKernel) = show(typeof(Kxp))
Base.show(io::IO, ::MIME"text/plain", Kxps::Vector{<:SplitKernel}) = println.(typeof.(Kxps))

function SplitKernel(x, cm::Cmap)
    s = size(x, 2)
    e = size(cm, 2)
    q = size(cm, 3)
    A = similar(x, e, q)
    B = similar(x, e, s)
    C = similar(x, s, q)
    return SplitKernel(cm, x, A, B, C)
end

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

function alloc_kernels(cov::ComposedKernel, x, xp::Cmap)
    Kxps = [SplitKernel(x, xp) for i in cov.kernels if i isa SquaredExp]
    return length(Kxps) == 1 ? Kxps[1] : Kxps
end

function alloc_kernels(cov::SquaredExp, x, xp::Cmap)
    return SplitKernel(x, xp)
end

function kernel(cov::ComposedKernel, hp, x, xp::Cmap)
    Kxps = alloc_kernels(cov, x, xp)
    kernel!(Kxps, cov, hp, x, xp)
    return Kxps
end

function kernel(cov::AbstractKernel, hp, x, xp::Cmap)
    Kxps = alloc_kernels(cov, x, xp)
    kernel!(Kxps, cov, hp, x, xp)
    return Kxps
end

function kernel!(Kxps::Vector{<:SplitKernel}, cov::ComposedKernel, hp, x, xp::Cmap)
    hps = split(hp, [dim_hp(t, size(x, 1)) for t in cov.kernels])
    Ks, hpn = rm_noise(cov, hps)
    for i in eachindex(Ks)
        kernel!(Kxps[i], Ks[i], hpn[i], x, xp)
    end
    return nothing
end

function kernel!(Kxp::SplitKernel, ::SquaredExp, hp, x, xp::Cmap)
    hps = copy(hp)
    hps[1] = 1.0
    kernel!(Kxp.A, SquaredExp(), hps, xp.xe, xp.xq; dist = SplitDistanceA())
    kernel!(Kxp.B, SquaredExp(), hps, xp.xe, x; dist = Euclidean())
    kernel!(Kxp.C, SquaredExp(), hp, x, xp.xq; dist = SplitDistanceC())
    return nothing
end

@inline kernel!(::WhiteNoise, Kxp::SplitKernel, hp, x, xp::Cmap) = nothing

@inline function kernel!(::AbstractKernel, Kxp::SplitKernel, hp, x, xp::Cmap)
    throw("Error: Covar not a SquaredExp")
end

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
