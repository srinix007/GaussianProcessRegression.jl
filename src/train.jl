struct TrainGPRCache{T,C<:AbstractKernel,P<:AbstractArray{T},X<:AbstractArray{T},
                     Y<:AbstractArray{T},K<:Vector{<:Matrix{T}},L<:Matrix{T}} <:
       AbstractModelCache
    cov::C
    hp::P
    x::X
    y::Y
    α::Y
    kerns::K
    kchol_base::L
    ∇K::L
    function TrainGPRCache(cov, hp, x, y, α, kerns, kchol, ∇K)
        return new{eltype(α),typeof(cov),typeof.((hp, x, y, kerns, kchol))...}(cov, hp, x,
                                                                               y, α, kerns,
                                                                               kchol, ∇K)
    end
end

function alloc_kernels(cov::AbstractKernel, x)
    n = size(x, 2)
    return [similar(x, n, n)]
end

function TrainGPRCache(cov::AbstractKernel, hp, x, y)
    n = size(x, 2)
    kerns = alloc_kernels(cov, x)
    kchol_base = similar(x, n, n)
    ∇K = similar(x, n, n)
    α = similar(x, n)
    return TrainGPRCache(cov, hp, x, y, α, kerns, kchol_base, ∇K)
end

@inline TrainGPRCache(md::AbstractGPRModel) = TrainGPRCache(md.covar, md.params, md.x, md.y)

function Base.show(io::IO, ::MIME"text/plain", tc::TrainGPRCache)
    println(typeof(tc))
    println("Cache of hp, x, y, α, kerns, kchol_base, ∇K")
    return nothing
end

function update_cache!(tc::TrainGPRCache, hp)
    tc.hp .= hp
    if tc.cov isa ComposedKernel
        kernels!(tc.kerns, tc.cov, hp, tc.x)
        tc.kchol_base .= tc.kerns[1]
        for t in 2:length(tc.kerns)
            lz(tc.kchol_base) .+= lz(tc.kerns[t])
        end
        add_noise!(tc.kchol_base, tc.cov, hp, tc.x)
    else
        kernel!(tc.kerns[1], tc.cov, hp, tc.x)
        tc.kchol_base .= tc.kerns[1]
    end
    kchol = cholesky!(tc.kchol_base)
    tc.α .= kchol \ tc.y
    return nothing
end

function loss(::MarginalLikelihood, mc::AbstractModelCache)
    kchol = Cholesky(UpperTriangular(mc.kchol_base))
    return loss(MarginalLikelihood(), kchol, mc.y, mc.α)
end

function grad(::MarginalLikelihood, mc::AbstractModelCache)
    ∇L = similar(mc.hp)
    kchol = Cholesky(UpperTriangular(mc.kchol_base))
    @inbounds for i in eachindex(∇L)
        ret = grad!(mc.cov, mc.∇K, i, mc.hp, mc.x, mc.kerns)
        ∇K = ret === nothing ? mc.∇K : ret
        ∇L[i] = grad(MarginalLikelihood(), kchol, ∇K, mc.α)
    end
    return ∇L
end

function train(md::AbstractModel; hp0=md.params, method=NelderMead())
    return G = similar(hp0)
end
