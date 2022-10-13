abstract type AbstractCache end

function Base.show(io::IO, ::MIME"text/plain", tc::AbstractCache)
    println(typeof(tc))
    println("Cache")
    for i in fieldnames(typeof(tc))
        if hasmethod(size, Tuple{fieldtype(typeof(tc), i)})
            println(i, " :: ", fieldtype(typeof(tc), i), " : ", size(getfield(tc, i)))
        else
            println(i, " :: ", fieldtype(typeof(tc), i))
        end
    end
    return nothing
end