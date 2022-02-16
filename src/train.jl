function init_params(::AbstractLoss, md::AbstractGPRModel)
    return rand(eltype(md.params), size(md.params))
end

function init_params(::MarginalLikelihood, md::AbstractGPRModel)
    return fill(0.1 * one(eltype(md.params)), size(md.params))
end

function train(md::AbstractModel, cost::AbstractLoss, hp0 = init_params(cost, md);
               method = ConjugateGradient(; linesearch = LineSearches.BackTracking()),
               options = Optim.Options())
    return train(method, md, cost, hp0, options)
end

function train(method::ZerothOrderOptimizer, md::AbstractModel, cost::AbstractLoss, hp0,
               options)
    tc = loss_cache(cost)(md)
    f = let tc = tc, cost = cost, md = md
        x -> loss(cost, x, md, tc)
    end
    return optimize(f, hp0, method, options)
end

function train(method::FirstOrderOptimizer, md::AbstractModel, cost::AbstractLoss, hp0,
               options)
    tc = loss_grad_cache(cost)(md)
    fg! = let tc = tc, cost = cost, md = md
        (F, G, x) -> loss_grad!(cost, F, G, x, md, tc)
    end
    return optimize(Optim.only_fg!(fg!), hp0, method, options)
end