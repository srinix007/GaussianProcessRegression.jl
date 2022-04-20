function kfoldcv(n, k, nb = div(n, k))
    nsh = shuffle(1:n)
    trn = Vector{Vector{typeof(n)}}()
    tst = Vector{Vector{typeof(n)}}()
    for i = 1:nb
        idx = (1+(i-1)*k):(i*k)
        push!(tst, nsh[idx])
        push!(trn, [nsh[j] for j in 1:n if !(j in idx)])
    end
    return (trn, tst)
end

function cv_batch(md::AbstractGPRModel, cost::AbstractLoss, x, y, cvset)
    trn, tst = cvset
    dim = size(x, 1)
    ntrn = length(trn[1])
    ntst = length(tst[1])
    xtst = similar(x, dim, ntst)
    xtrn = similar(x, dim, ntrn)
    ytst = similar(y, ntst)
    ytrn = similar(y, ntrn)
    yp = similar(y, ntst)
    Σp = similar(yp, ntst, ntst)
    lss = similar(y, length(trn))
    mdt = similar(md, md.params, rand(eltype(x), dim, ntrn), rand(eltype(y), ntrn))
    pc = predict_cache(mdt, xtst)(mdt, ntst)
    for i in eachindex(trn)
        @views xtst .= x[:, tst[i]]
        @views ytst .= y[tst[i]]
        @views mdt.x .= x[:, trn[i]]
        @views mdt.y .= y[trn[i]]
        lss[i] = cv_step!(cost, mdt, xtst, ytst, pc, yp, Σp)
    end
    return lss
end

function cv_step(md::AbstractGPRModel, cost::AbstractLoss, xtr, ytr, xtst, ytst)
    yp = similar(ytst)
    Σp = similar(yp, size(yp, 1), size(yp, 1))
    mdt = similar(md, md.params, xtr, ytr)
    pc = predict_cache(mdt, xtst)(mdt, xtst)
    return cv_step!(cost, mdt, xtst, ytst, pc, yp, Σp)
end

function cv_step!(cost::AbstractLoss, mdt::AbstractGPRModel, xtst, ytst,
                  pc::AbstractPredictCache, yp, Σp)
    update_cache!(pc, mdt)
    predict!(yp, Σp, mdt, xtst, pc)
    return loss(cost, ytst, yp, Σp)
end
