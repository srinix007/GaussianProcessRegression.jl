using Documenter, GaussianProcessRegression

push!(LOAD_PATH, "../src/")

makedocs(; sitename="GaussianProcessRegression.jl",
         pages=["Introduction" => "index.md", "Sampling" => "sampling.md"])
