abstract type AbstractCache end

function Base.show(io::IO, ::MIME"text/plain", tc::AbstractCache)
    println(typeof(tc))
    println("Loss cache")
    for i in fieldnames(typeof(tc))
        println(i, " :: ", fieldtype(typeof(tc), i), " : ", size(getfield(tc, i)))
    end
    return nothing
end