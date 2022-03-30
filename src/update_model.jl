abstract type AbstractUpdater end
struct BFGSQuad <: AbstractUpdater end

updater_cache(::BFGSQuad) = BFGSQuadCache

## High-level API

function update_sample!(md::AbstractGPRModel, δy, upd::AbstractUpdater, cost::AbstractLoss,
                        ϵJ = 1e-3)
    tc = grad_cache(cost)(md)
    uc = updater_cache(upd)(md)
    iters = update_sample!(md, δy, cost, uc, tc, ϵJ)
    return iters
end

## Low-level API

function update_cache!(uc::BFGSQuadCache, md::AbstractGPRModel, cost::AbstractLoss,
                       tc::AbstractGradCache)
    uc.hp .= log.(md.params)
    jac = let md = md, cost = cost, tc = tc
        function jj(log_x)
            G = similar(log_x)
            log_loss_grad!(cost, nothing, G, log_x, md, tc)
            return G
        end
    end
    uc.J .= jac(uc.hp)
    hess = hessian_fd(jac, uc.hp)
    uc.hess_inv .= inv(Hermitian(hess))
    return nothing
end

function update_sample!(md::AbstractGPRModel, δy, cost::AbstractLoss,
                        uc::AbstractUpdateCache, tc::AbstractGradCache, ϵJ = 1e-3)
    md.y .+= δy
    update_cache!(uc, md, cost, tc)
    jac = let md = md, cost = cost, tc = tc
        function jj(x)
            G = similar(x)
            log_loss_grad!(cost, nothing, G, x, md, tc)
            return G
        end
    end
    iters = bfgs_quad!(uc.hp, uc.J, uc.hess_inv, jac, ϵJ)
    md.params .= exp.(uc.hp)
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
