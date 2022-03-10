function init_params(::AbstractLoss, md::AbstractGPRModel)
    return rand(eltype(md.params), size(md.params))
end

function init_params(::MarginalLikelihood, md::AbstractGPRModel)
    return ones(eltype(md.params), size(md.params))
end

function train(md::AbstractModel, cost::AbstractLoss, hp0 = init_params(cost, md);
               method = ConjugateGradient(), options = Optim.Options())
    return train(islog(cost, md), method, md, cost, hp0, options)
end

function train(::NoLogScale, method::Optim.ZerothOrderOptimizer, md::AbstractModel,
               cost::AbstractLoss, hp0, options)
    tc = loss_cache(cost)(md)
    f = let tc = tc, cost = cost, md = md
        x -> loss(cost, x, md, tc)
    end
    res = optimize(f, hp0, method, options)
    hpmin = Optim.minimizer(res)
    return hpmin, res
end

function train(::LogScale, method::Optim.ZerothOrderOptimizer, md::AbstractModel,
               cost::AbstractLoss, hp0, options)
    tc = loss_cache(cost)(md)
    f = let tc = tc, cost = cost, md = md
        x -> loss(cost, exp.(x), md, tc)
    end
    res = optimize(f, hp0, method, options)
    log_hpmin = Optim.minimizer(res)
    return exp.(log_hpmin), res
end

function train(::NoLogScale, method::Optim.FirstOrderOptimizer, md::AbstractModel,
               cost::AbstractLoss, hp0, options)
    tc = loss_grad_cache(cost)(md)
    fg! = let tc = tc, cost = cost, md = md
        (F, G, x) -> loss_grad!(cost, F, G, x, md, tc)
    end
    res = optimize(Optim.only_fg!(fg!), hp0, method, options)
    hpmin = Optim.minimizer(res)
    return hpmin, res
end

function train(::LogScale, method::Optim.FirstOrderOptimizer, md::AbstractModel,
               cost::AbstractLoss, hp0, options)
    tc = loss_grad_cache(cost)(md)
    fg! = let tc = tc, cost = cost, md = md
        (F, G, x) -> log_loss_grad!(cost, F, G, x, md, tc)
    end
    res = optimize(Optim.only_fg!(fg!), hp0, method, options)
    log_hpmin = Optim.minimizer(res)
    return exp.(log_hpmin), res
end

function train(::NoLogScale, method::Optim.SecondOrderOptimizer, md::AbstractModel,
               cost::AbstractLoss, hp0, options)
    tc = loss_grad_cache(cost)(md)
    f = let tc = tc, cost = cost, md = md
        x -> loss(cost, x, md, tc)
    end
    g! = let tc = tc, cost = cost, md = md
        (G, x) -> grad!(G, cost, x, md, tc)
    end
    res = optimize(f, g!, hp0, method, options)
    hpmin = Optim.minimizer(res)
    return hpmin, res
end

function train(::LogScale, method::Optim.SecondOrderOptimizer, md::AbstractModel,
               cost::AbstractLoss, hp0, options)
    tc = loss_grad_cache(cost)(md)
    f = let tc = tc, cost = cost, md = md
        x -> loss(cost, exp.(x), md, tc)
    end
    g! = let tc = tc, cost = cost, md = md
        (G, x) -> begin
            grad!(G, cost, exp.(x), md, tc)
            G .*= exp.(x)
        end
    end
    res = optimize(f, g!, hp0, method, options)
    log_hpmin = Optim.minimizer(res)
    return exp.(log_hpmin), res
end
