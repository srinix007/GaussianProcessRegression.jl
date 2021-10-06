
function update!(md::GPRModel)
    if md.info[:init] === false
        kernel!(md.Kxx, md.covar, md.params, md.x)
        lkxx = cholesky!(md.Kxx)
        ldiv!(md.wt, lkxx, md.y)
    end
    return nothing
end

function predict(md::GPRModel{C, T, P, X}, xp::X ) where {C,T,P,X}
    yp = similar(xp, size(xp,2))
    predict!(yp, md, xp)
    return yp
end

function predict!(yp, md::GPRModel{C, T, P, X}, xp::X) where {C,T,P,X}
    Kxp = kernel(md.covar, md.params, xp, md.x)
    update!(md)
    mul!(yp, Kxp, md.wt)
    return nothing
end