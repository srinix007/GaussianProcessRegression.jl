### A Pluto.jl notebook ###
# v0.19.10

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

# ╔═╡ f2ff2329-537f-4f8e-984d-e60f9f69ba8d
xp = reshape(collect(0:0.1:10), (1,:));

# ╔═╡ 25cb4d6f-d3e9-4969-9931-d6834a66f299
cov = SquaredExp()

# ╔═╡ d94133a5-94a8-4723-ba40-9f154fa60374
gpp = GaussianProcess(x->zero(eltype(x)), cov)

# ╔═╡ ae4fd00b-2cde-428e-8534-17d7d25254d8
#res = let
	#f(x) = loss(MarginalLikelihood(), exp.(x), md1)
	#J(x) = x .* grad(MarginalLikelihood(), exp.(x), md1)
	#res = optimize(f, [0.1, 0.1], LBFGS(), Optim.Options(g_tol=1e-3); inplace=false)
	#res
#end

# ╔═╡ 6d7ab6ca-0b06-4a76-a1af-468bd858205b
NelderMead <: Optim.ZerothOrderOptimizer

# ╔═╡ 68d6be64-4d07-4736-a2c6-17e5b9eb0bbf
@bind σ Slider(0.01:0.1:10)

# ╔═╡ 05c617fc-5b43-4fe5-8f93-d45c6faca3ed
md"σ = $σ"

# ╔═╡ e920f7c2-5301-46e2-bb39-46dd5a5aa7ac
@bind l Slider(0:0.1:2)

# ╔═╡ f1beb05a-7a93-4271-8f5f-2d39092c5eab
θ = [σ, l]

# ╔═╡ 3f869f8a-98a6-4a13-9a76-3cf18872cff2
ypp = sample(gpp, xp, θ);

# ╔═╡ 3aac3fbb-2c90-4e54-b537-128c85b755c9
yp = copy(ypp);

# ╔═╡ 1f74cb05-4fa6-423b-9d61-9443f6e6e151
md"l = $l"

# ╔═╡ 6db587f9-d626-44bf-a591-106acc740ab6
@bind n Slider(5:50)

# ╔═╡ 0c1d5bc4-d14c-4f54-925a-6373550b5338
inds = randperm(100)[1:n]

# ╔═╡ 58c511e1-7339-4bde-a1bf-792a5082e593
x, y= reshape(xp[inds], (1,:)), yp[inds];

# ╔═╡ 59cd0a20-1d26-4672-8a6f-62dc2ac56ea0
md1 = GPRModel(cov, x, y);

# ╔═╡ d784f186-c2c9-47fb-82a0-6c6daa5a55e5
md1.params, loss(MarginalLikelihood(), md1)

# ╔═╡ b5cc801c-b032-465a-9fc3-3df531459e91
let
	log_ls = -3.0:0.1:1.0
	ls = exp.(log_ls)
	σ = 1.0
	lss = [loss(MarginalLikelihood(), [σ, l], md1) for l in ls]
	grads = [grad(MarginalLikelihood(), [σ, l], md1)[2] for l in ls]
	p = plot()
	plot!(p, log_ls, lss)
	plot!(p, log_ls, abs.(ls .* grads))
end

# ╔═╡ a02c352f-4fe8-438d-8bf8-eab8e95fe2a8
hpmin, res = train(md1, MarginalLikelihood(); options = Optim.Options(g_tol=1e-2))

# ╔═╡ f9e08302-28a9-4fc3-8bc3-296e0bc99d5c
loss(MarginalLikelihood(), hpmin, md1)

# ╔═╡ f70d0308-c817-40a2-8135-be1f506d5ed1
md1_opt = GPRModel(cov, hpmin, x, y);

# ╔═╡ 7c564648-1e35-4e89-b4b9-e9be46877382
hpmin ≈ md1_opt.params

# ╔═╡ d3a800bb-0791-474b-8354-8e4f8ed618ce
loss(MarginalLikelihood(), md1_opt)

# ╔═╡ f4eaadf6-4836-4f6f-8022-547aaf8dc1bc
ypred, Σpred = predict(md1_opt, xp);

# ╔═╡ 2679c2ac-d840-4242-921d-937772919fa6
md"n = $n"

# ╔═╡ eafb9f81-7dd3-4797-8267-9c0bbd55fbd6
let 
	p = plot(legend=:topleft)
	scatter!(p, x[1,:], y, color="black", label="samples")
	plot!(p, xp[1,:], yp, xlabel=L"x", ylabel=L"f(x)", label=L"f(x)")
	plot!(p, xp[1,:], ypred, ribbon=2 .* sqrt.(diag(Σpred)), label=L"GPR(f, x)")
	savefig(p, "GPR_1d.pdf")
	p
end

# ╔═╡ dae0a2fa-b6ce-46ad-9d70-cadf94aadf0f
yp2 = sample(gpp, xp, θ);

# ╔═╡ 339766e9-dbab-4758-97ea-c24c23de506c
yp_sum = yp .+ 2e-1 .* yp2;

# ╔═╡ d0b96121-dad5-4384-a760-014f2632baa1
let 
	p = plot(xlabel=L"x", ylabel=L"f(x)", legend=:outertopright)
	plot!(xp[1,:], yp, label=L"f(x)")
	plot!(xp[1,:], yp2, label=L"\delta h(x)")
	plot!(xp[1,:], yp_sum, label=L"f'(x)")
	min1 = argmin(yp)
	min2 = argmin(yp_sum)
	xmin, ypmin = xp[1,min1], yp[min1]
	xmin1, ypmin1 = xp[1,min2], yp_sum[min2]
	scatter!(p, [xmin], [ypmin], color="blue", label=L"min(f(x))")
	scatter!(p, [xmin1], [ypmin1], color="green", label=L"min(f'(x))")
	vline!(p, [xmin], color="blue", label="")
	vline!(p, [xmin1], color= "green", label="")
	savefig(p, "Minima_shift.pdf")
	p
end

# ╔═╡ 54b42d4c-6293-4a9a-9e5c-6178ee46af45


# ╔═╡ Cell order:
# ╠═a09e1670-45de-11ec-0a4b-9d1ced6b0199
# ╠═72620701-24dd-460e-8b49-0c53dcdd577f
# ╠═f2ff2329-537f-4f8e-984d-e60f9f69ba8d
# ╠═f1beb05a-7a93-4271-8f5f-2d39092c5eab
# ╠═25cb4d6f-d3e9-4969-9931-d6834a66f299
# ╠═d94133a5-94a8-4723-ba40-9f154fa60374
# ╠═3f869f8a-98a6-4a13-9a76-3cf18872cff2
# ╠═3aac3fbb-2c90-4e54-b537-128c85b755c9
# ╠═0c1d5bc4-d14c-4f54-925a-6373550b5338
# ╠═58c511e1-7339-4bde-a1bf-792a5082e593
# ╠═59cd0a20-1d26-4672-8a6f-62dc2ac56ea0
# ╠═d784f186-c2c9-47fb-82a0-6c6daa5a55e5
# ╠═b5cc801c-b032-465a-9fc3-3df531459e91
# ╠═ae4fd00b-2cde-428e-8534-17d7d25254d8
# ╠═6d7ab6ca-0b06-4a76-a1af-468bd858205b
# ╠═a02c352f-4fe8-438d-8bf8-eab8e95fe2a8
# ╠═f9e08302-28a9-4fc3-8bc3-296e0bc99d5c
# ╠═f70d0308-c817-40a2-8135-be1f506d5ed1
# ╠═7c564648-1e35-4e89-b4b9-e9be46877382
# ╠═d3a800bb-0791-474b-8354-8e4f8ed618ce
# ╠═f4eaadf6-4836-4f6f-8022-547aaf8dc1bc
# ╟─05c617fc-5b43-4fe5-8f93-d45c6faca3ed
# ╟─68d6be64-4d07-4736-a2c6-17e5b9eb0bbf
# ╟─1f74cb05-4fa6-423b-9d61-9443f6e6e151
# ╟─e920f7c2-5301-46e2-bb39-46dd5a5aa7ac
# ╟─2679c2ac-d840-4242-921d-937772919fa6
# ╠═6db587f9-d626-44bf-a591-106acc740ab6
# ╟─eafb9f81-7dd3-4797-8267-9c0bbd55fbd6
# ╠═dae0a2fa-b6ce-46ad-9d70-cadf94aadf0f
# ╠═339766e9-dbab-4758-97ea-c24c23de506c
# ╟─d0b96121-dad5-4384-a760-014f2632baa1
# ╠═54b42d4c-6293-4a9a-9e5c-6178ee46af45
