function Base.:+(::T, ::S) where {T <: AbstractKernel,S <: AbstractKernel}
    return [T(), S()]
end

function Base.:+(K::Vector{<:AbstractKernel}, ::S) where {S <: AbstractKernel}
    return vcat(K, S())
end

function Base.split(A::AbstractVector, inds)
    cind = vcat(0, cumsum(inds))
    return [A[i] for i in (:).(cind[1:end - 1] .+ 1, cind[2:end])]
end

function dim_hp(K::Vector{<:AbstractKernel}, dim)
    return sum(dim_hp(k, dim) for k in K)
end

function rm_noise(K::Vector{<:AbstractKernel}, hp) 
    ninds = findall(x -> x === WhiteNoise(), K)
    hpn = length(ninds) > 0 ? hp[1:end - 1] : hp
    return deleteat!(copy(K), ninds), hpn
end

function kernel(K::Vector{<:AbstractKernel}, hp, x, xp)
    Ks, hpn = rm_noise(K, hp)
    dim = first(size(x))
    hps = split(hpn, [dim_hp(t, dim) for t in Ks])

    if length(Ks) > 1
        kern = kernel(Ks[1], hps[1], x, xp)
        for t in 2:length(Ks)
            kern .+= kernel(Ks[t], hps[t], x, xp)
        end
        # kern = mapreduce(t -> kernel(Ks[t], hps[t], x, xp), (x, y) -> pev(lz(x) .+ lz(y)), eachindex(Ks))
    else
        kern = kernel(Ks[1], hpn, x, xp)
    end

    return kern
end

function kernel(K::Vector{<:AbstractKernel}, hp, x)
    kern = kernel(K, hp, x, x)
     if WhiteNoise() in K
        kern[diagind(kern)] .+= (hp[end]^2)
     end
     return kern
end

