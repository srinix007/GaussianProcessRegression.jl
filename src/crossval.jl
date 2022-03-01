function kfoldcv(n, k)
    l = div(n, k)
    nsh = shuffle(1:n)
    trn = Vector{Vector{typeof(n)}}()
    tst = Vector{Vector{typeof(n)}}()
    for i = 1:l
        idx = (1+(i-1)*k):(i*k)
        push!(tst, nsh[idx])
        push!(trn, [nsh[j] for j in 1:n if !(j in idx)])
    end
    return (trn, tst)
end

function cv_batch(md::AbstractGPRModel, cost::AbstractLoss, x, y, cvset, nbatch)
    trn, tst = cvset
    ntrn = length(trn[1])
    ntst = length(tst[1])
    xtst = similar(x, size(x, 1), ntst)
    xtrn = similar(x, size(x, 1), ntrn)
    ytst = similar(y, ntst)
    ytrn = similar(y, ntrn)
    yp = similar(y, ntst)
    Σp = similar(yp, ntst, ntst)
    pc = predict_cache(md)(md, ntst)
    lss = similar(y, nbatch)
    for i = 1:nbatch
        xtst .= x[:, tst[i]]
        xtrn .= x[:, trn[i]]
        ytst .= y[tst[i]]
        ytrn .= y[trn[i]]
        lss[i] = cv_step!(md, cost, xtr, ytr, xtst, ytst, pc, yp, Σp)
    end
    return lss
end

function cv_step(md::AbstractGPRModel, cost::AbstractLoss, xtr, ytr, xtst, ytst)
    yp = similar(ytst)
    Σp = similar(yp, size(yp, 1), size(yp, 1))
    pc = predict_cache(md)(md, size(xtst, 2))
    return cv_step!(md, cost, xtr, ytr, xtst, ytst, pc, yp, Σp)
end

function cv_step!(md::AbstractGPRModel, cost::AbstractLoss, xtr, ytr, xtst, ytst,
                  pc::AbstractPredictCache, yp, Σp)
    mdt = typeof(md)(md.covar, md.params, xtr, ytr)
    predict!(yp, Σp, mdt, xtst, pc)
    return loss(cost, yp, Σp, ytst)
end
