abstract type AbstractUpdater end
struct BFGSQuad <: AbstractUpdater end

updater_cache(::BFGSQuad) = BFGSQuadCache

## High-level API

function update_sample!(md::AbstractGPRModel, δy, upd::AbstractUpdater, cost::AbstractLoss)
    tc = grad_cache(cost)(md)
    uc = updater_cache(upd)(md)
    update_cache!(uc, md, cost)
    iters = update_sample!(md, δy, cost, uc, tc)
    return iters
end

## Low-level API

function update_cache!(uc::BFGSQuadCache, md::AbstractGPRModel, cost::AbstractLoss)
    uc.hp .= md.params
    tc = grad_cache(cost)(md)
    grad!(uc.J, cost, md.params, md, tc)
    log_jac = let md = md, cost = cost, tc = tc
        function jj(log_x)
            ret = similar(log_x)
            x = exp.(log_x)
            grad!(ret, cost, x, md, tc)
            return ret .* x
        end
    end
    hessian_fd!(uc.hess, log_jac, log.(md.params))
    return nothing
end

function update_sample!(md::AbstractGPRModel, δy, cost::AbstractLoss,
                        uc::AbstractUpdateCache, tc::AbstractGradCache)
    md.y .+= δy
    log_jac = let md = md, cost = cost, tc = tc
        function jj(log_x)
            ret = similar(log_x)
            x = exp.(log_x)
            grad!(ret, cost, x, md, tc)
            return ret .* x
        end
    end
    log_hp = log.(uc.hp)
    iters = bfgs_quad!(log_hp, uc.J, uc.hess, log_jac, uc.ϵJ)
    md.params .= exp.(log_hp)
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
