### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ a09e1670-45de-11ec-0a4b-9d1ced6b0199
begin
	using Pkg
	Pkg.activate("./GaussianProcessRegression.jl/")
	using Plots
	using Optim
	using LineSearches
	using Revise
	using LinearAlgebra
	using PlutoUI
	using Random
	using LaTeXStrings
end

# ╔═╡ 72620701-24dd-460e-8b49-0c53dcdd577f
using GaussianProcessRegression

# ╔═╡ a73447e3-4683-4968-874a-c52455d8b55b
xp = reshape(collect(0:0.01:1), (1,:))

# ╔═╡ e3d81694-1113-4135-862f-d801e3d029eb
cov = SquaredExp()

# ╔═╡ 75b9062b-72b7-4843-801a-969a1e688dc5
gpp = GaussianProcess(x->zero(eltype(x)), cov)

# ╔═╡ b6e5ed70-6c56-4f1a-a3e6-b02c2e0d5e57
@bind σ Slider(0.1:0.1:2)

# ╔═╡ c022b7da-42a5-401d-84c6-5a304106948f
md"σ=$σ"

# ╔═╡ 8f68a881-b38e-4fa1-a30c-16e223997b55
@bind l Slider(0.1:0.1:10)

# ╔═╡ f327531a-b236-492a-b3a4-f84a4bcb9be6
θ = [σ, l]

# ╔═╡ 4d762f33-d016-4a18-9afb-da9043a7c70e
yp = sample(gpp, xp, θ);

# ╔═╡ 14150fb5-314c-465e-9213-0d2766903091
md"l=$l"

# ╔═╡ e862d44d-93fc-4eff-a566-ec3cd4fb37bf
@bind n Slider(5:2:50)

# ╔═╡ 7e2f6540-7f4e-442c-a215-81db10c02b6e
md"n=$n"

# ╔═╡ 486bd41e-f206-454a-861b-2af1ffecbf1c
inds = randperm(100)[1:n];

# ╔═╡ 0f1ea937-11c2-4938-83ce-d4d47a8aff02
x, y= reshape(xp[inds], (1,:)), yp[inds];

# ╔═╡ 836262e2-9c42-4383-84a7-411c75e3e817
mdd = GPRModel(cov, x, y)

# ╔═╡ cdfe987a-660f-40b0-ad75-de057d8f5b1a
mdd.params

# ╔═╡ 73497f30-6111-475e-b219-be3b2698978c
loss(MarginalLikelihood(), mdd)

# ╔═╡ 150cad2d-76f1-4e73-88d8-19e1a9dac6e0
grad(MarginalLikelihood(), mdd)

# ╔═╡ 1ebebf59-e093-4583-af06-78c8d8282c83
let
	log_ls = 0.1:0.1:2
	ls = exp.(log_ls)
	σ = 1.1
	lss = [loss(MarginalLikelihood(), [σ, l], mdd) for l in ls]
	grads = [grad(MarginalLikelihood(), [σ, l], mdd)[2] for l in ls]
	p = plot(; legend=:outertopright)
	plot!(p, log_ls, lss, label="loss")
	plot!(p, log_ls, ls .* grads, label="log_grad")
	plot!(p, log_ls, grads, label="grad")
end

# ╔═╡ 05ffd8c4-f7b4-4a1b-b5ac-e1b85b1819fc
let
	f(x) = loss(MarginalLikelihood(), exp.(x), mdd)
	g(x) = exp.(x) .* grad(MarginalLikelihood(), exp.(x), mdd)
	x0 = [1.0,1.0]
	options = Optim.Options(;g_tol=1e-2)
	res = optimize(f, g, x0, Newton(), options ;  inplace=false)
	hpmin = Optim.minimizer(res)
	exp.(hpmin), res
end

# ╔═╡ 650f8ced-e98d-4d37-bb74-55f8a42f989d
exp()

# ╔═╡ 25ca46d6-18e2-4703-a904-c489a86b8ea9
hpmin, res = train(LogScale(), ConjugateGradient(), mdd, MarginalLikelihood(), [0.1,0.1], Optim.Options(;g_tol=1e-2))

# ╔═╡ cfbcf0be-6c9c-4b1d-83a2-222ae2282223
loss(MarginalLikelihood(), hpmin, mdd)

# ╔═╡ 8ce37e05-9104-4091-b3d1-0e8993873d82
norm(grad(MarginalLikelihood(), hpmin, mdd))

# ╔═╡ 1af1707f-4fea-4eda-8065-8b5885a4a796
let
	xpp = xp[1,:]
	p = plot(xpp, yp)
	scatter!(p, x[1,:], y)
end

# ╔═╡ Cell order:
# ╠═a09e1670-45de-11ec-0a4b-9d1ced6b0199
# ╠═72620701-24dd-460e-8b49-0c53dcdd577f
# ╠═a73447e3-4683-4968-874a-c52455d8b55b
# ╠═f327531a-b236-492a-b3a4-f84a4bcb9be6
# ╠═e3d81694-1113-4135-862f-d801e3d029eb
# ╠═75b9062b-72b7-4843-801a-969a1e688dc5
# ╠═4d762f33-d016-4a18-9afb-da9043a7c70e
# ╟─c022b7da-42a5-401d-84c6-5a304106948f
# ╟─b6e5ed70-6c56-4f1a-a3e6-b02c2e0d5e57
# ╟─14150fb5-314c-465e-9213-0d2766903091
# ╟─8f68a881-b38e-4fa1-a30c-16e223997b55
# ╟─7e2f6540-7f4e-442c-a215-81db10c02b6e
# ╠═e862d44d-93fc-4eff-a566-ec3cd4fb37bf
# ╠═486bd41e-f206-454a-861b-2af1ffecbf1c
# ╠═0f1ea937-11c2-4938-83ce-d4d47a8aff02
# ╠═836262e2-9c42-4383-84a7-411c75e3e817
# ╠═cdfe987a-660f-40b0-ad75-de057d8f5b1a
# ╠═73497f30-6111-475e-b219-be3b2698978c
# ╠═150cad2d-76f1-4e73-88d8-19e1a9dac6e0
# ╠═1ebebf59-e093-4583-af06-78c8d8282c83
# ╠═05ffd8c4-f7b4-4a1b-b5ac-e1b85b1819fc
# ╠═650f8ced-e98d-4d37-bb74-55f8a42f989d
# ╠═25ca46d6-18e2-4703-a904-c489a86b8ea9
# ╠═cfbcf0be-6c9c-4b1d-83a2-222ae2282223
# ╠═8ce37e05-9104-4091-b3d1-0e8993873d82
# ╟─1af1707f-4fea-4eda-8065-8b5885a4a796
