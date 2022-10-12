abstract type AbstractCostCache <: AbstractCache end
abstract type AbstractLossCache <: AbstractCostCache end
abstract type AbstractGradCache <: AbstractCostCache end
abstract type AbstractLossGradCache <: AbstractCostCache end

struct MllLossCache{T,P<:AbstractArray{T},L<:AbstractArray{T},Y<:AbstractArray{T}} <:
       AbstractLossCache
    hp::P
    kchol_base::L
    α::Y
end

function MllLossCache(md::AbstractGPRModel)
    nx = size(md.x, 2)
    K = similar(md.x, nx, nx)
    hp = copy(md.params)
    α = similar(md.y, size(md.y, 1))
    MllLossCache{eltype(md.x),typeof(hp),typeof(K),typeof(α)}(hp, K, α)
end

struct MllGradCache{T,P<:AbstractArray{T},Y<:AbstractArray{T},K<:Vector{<:Matrix{T}},
    L<:Matrix{T}} <: AbstractGradCache
    hp::P
    α::Y
    kerns::K
    kchol_base::L
    ∇K::L
    K⁻¹::L
    tt::Y
end


function MllGradCache(md)
    nx = size(md.x, 2)
    kchol_base = similar(md.x, nx, nx)
    hp = copy(md.params)
    α = similar(md.y, size(md.y, 1))
    kerns = alloc_kernels(md.covar, md.x)
    ∇K = similar(kchol_base)
    K⁻¹ = similar(kchol_base)
    tt = similar(α)
    MllGradCache{eltype(α),typeof.((hp, α, kerns, kchol_base))...}(hp, α, kerns, kchol_base,
        ∇K, K⁻¹, tt)
end
