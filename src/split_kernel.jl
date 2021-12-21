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

Base.size(Kxp::SplitKernel) = (size(Kxp.x, 2), size(Kxp.A)...)
Base.size(Kxp::SplitKernel, i) = size(Kxp)[i]

function Base.getindex(Kxp::SplitKernel, ::Colon, ::Colon, ::Colon)
    sr = 1:size(Kxp, 1)
    er = 1:size(Kxp, 2)
    qr = 1:size(Kxp, 3)
    nr = 1:size(Kxp, 4)
    KK = zeros(eltype(Kxp.A), length(sr), length(er), length(qr))
    @turbo for n in nr, q in qr, e in er, s in sr
        KK[s, e, q] += Kxp.A[e, q, n] * Kxp.B[e, s, n] * Kxp.C[s, q, n]
    end
    return KK
end

function Base.getindex(Kxp::SplitKernel, ::Colon, e::Int, q::Int)
    sr = 1:size(Kxp, 1)
    nr = 1:size(Kxp, 4)
    KK = zeros(eltype(Kxp.A), length(sr))
    @turbo for n in nr, s in sr
        KK[s] += Kxp.A[e, q, n] * Kxp.B[e, s, n] * Kxp.C[s, q, n]
    end
    return KK
end

function Base.getindex(Kxp::SplitKernel, s::Int, e::Int, q::Int)
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

function kernel(cov::ComposedKernel, hp, x, xp::Cmap)
    Kxps = SplitKernel(cov, x, xp)
    kernel!(Kxps, cov, hp, x, xp)
    return Kxps
end

function kernel(::SquaredExp, hp, x, xp::Cmap)
    Kxps = SplitKernel(SquaredExp(), x, xp)
    kernel!(Kxps, SquaredExp(), hp, x, xp)
    return Kxps
end

function kernel!(Kxps::SplitKernel, cov::ComposedKernel, hp, x, xp::Cmap)
    hps = split(hp, [dim_hp(t, size(x, 1)) for t in cov.kernels])
    Ks, hpn = rm_noise(cov, hps)
    for i in eachindex(Ks)
        kernel!(Kxps, i, Ks[i], hpn[i], x, xp)
    end
    return nothing
end

function kernel!(Kxps::SplitKernel, ::SquaredExp, hp, x, xp::Cmap)
    kernel!(Kxps, 1, SquaredExp(), hp, x, xp)
    return nothing
end


function kernel!(Kxp::SplitKernel, nkrn::Int, ::SquaredExp, hp, x, xp::Cmap)
    hps = copy(hp)
    hps[1] = 1.0
    @views kernel!(Kxp.A[:, :, nkrn], SquaredExp(), hps, xp.xe, xp.xq;
                   dist = SplitDistanceA())
    @views kernel!(Kxp.B[:, :, nkrn], SquaredExp(), hps, xp.xe, x; dist = Euclidean())
    @views kernel!(Kxp.C[:, :, nkrn], SquaredExp(), hp, x, xp.xq; dist = SplitDistanceC())
    return nothing
end

#@inline kernel!(::WhiteNoise, Kxp::SplitKernel, hp, x, xp::Cmap) = nothing

#@inline function kernel!(::AbstractKernel, Kxp::SplitKernel, hp, x, xp::Cmap)
#    throw("Error: Covar not a SquaredExp")
#end

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
