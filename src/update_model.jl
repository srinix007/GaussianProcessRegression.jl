struct BFGSQuadCache{T,K<:AbstractArray{T},W<:AbstractArray{T}} <: AbstractModelCache
    J::K
    hess::W
    ϵJ::T
    function BFGSQuadCache(J, hess, ϵJ)
        return new{eltype(J),typeof.((J, hess))...}(J, hess, ϵJ)
    end
end

function BFGSQuadCache(md::AbstractGPRModel, cost::AbstractLoss, hp)
    np = size(hp, 1)
    J = grad(cost, md)
    jac = let cost = cost, md = md
        x -> grad(cost, x, md)
    end
    hess = hessian_fd(jac, hp)
    return BFGSQuadCache(J, hess, norm(J))
end

function update_sample!(md::AbstractGPRModel, cost::AbstractLoss, δy)
    md.y .+= δy
    tc = model_cache(md)(md)
    uc = BFGSQuadCache(md, cost, md.params)
    iters = update_sample!(uc, tc, cost)
    return iters
end

function update_sample!(mc::BFGSQuadCache, tc::AbstractModelCache, cost::AbstractLoss)
    function jac(x)
        ret = similar(x)
        let tc = tc
            update_cache!(tc, x)
            ret = grad(cost, tc)
        end
        return ret
    end
    iters = bfgs_quad!(md.hp, mc.J, mc.hess, jac, mc.ϵJ)
    return iters
end

function bfgs_hessian(Bi, s, t, ρ = one(eltype(s)) / dot(s, t))
    C = I - ρ * s * t'
    B = C * Bi * C' + ρ * s * s'
    return Hermitian(B)
end

function bfgs_quad(xx, JJ, HH, jac::Function; ϵ = 1e-5, max_iter = 100)
    x0 = copy(xx)
    J0 = copy(JJ)
    B = inv(HH) + zeros(eltype(xx), length(xx), length(xx))
    iters = bfgs_quad!(x0, J0, B, jac, ϵ, max_iter)
    return x0, J0, inv(B), iters
end

function bfgs_quad!(θ, JJ, B, ∇L, ϵ, max_iter = 100)
    s = similar(θ)
    t = similar(s)
    iter = 0
    while norm(JJ) > ϵ && iter < max_iter
        s .= θ
        t .= JJ
        θ .= θ .- B * JJ
        JJ .= ∇L(θ)
        s .= θ .- s
        t .= JJ .- t
        B .= bfgs_hessian(B, s, t)
        iter += 1
    end
    return iter
end

function hessian_fd(∇L, x, ϵ = 1e-6)
    hess = similar(x, length(x), length(x))
    hessian_fd!(hess, ∇L, x, ϵ)
    return hess
end

function hessian_fd!(hess, ∇L::Function, x, ϵ = 1e-6)
    for i in eachindex(x)
        x_ϵ = copy(x)
        x_ϵ[i] += ϵ
        hess[:, i] .= (∇L(x_ϵ) - ∇L(x)) ./ ϵ
    end
    return hess
end
