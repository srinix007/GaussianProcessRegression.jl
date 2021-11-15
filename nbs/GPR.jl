### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ a09e1670-45de-11ec-0a4b-9d1ced6b0199
begin
	using Pkg
	Pkg.activate("./GaussianProcessRegression.jl/")
	using Plots
	using Optim
	using Revise
	using LinearAlgebra
end

# ╔═╡ 72620701-24dd-460e-8b49-0c53dcdd577f
using GaussianProcessRegression

# ╔═╡ a0895e98-2d01-48ae-b249-22655a21b71c
x  = rand(1, 5);

# ╔═╡ a786390d-fc31-4eed-9d97-1bf29485e2e6
f(x) = dropdims(sum((@. 25.0 * sin(2.0*cos(3.0*x))), dims=1), dims=1);

# ╔═╡ 13c44ff7-c021-4ae7-a5b2-3f6a49a864c5
y = f(x)

# ╔═╡ 68f21ff2-f6d8-4132-ac1e-978d9be87d94
md = GPRModel(SquaredExp(), x, y)

# ╔═╡ 9bc25e84-35d5-4e7f-8c26-81ecdfb12b7a
res = train(md, MarginalLikelihood())

# ╔═╡ deb35813-a714-4aed-8ef3-72d008f591f1
begin
	hp0 = md.params
	hpmin = Optim.minimizer(res)
end

# ╔═╡ 823cb765-acb8-439c-8197-44f3cb9ebd41
hp0, hpmin

# ╔═╡ ce861ed4-f98a-4052-bebe-590c15cc259b
loss(MarginalLikelihood(), md), loss(MarginalLikelihood(), hpmin, md)

# ╔═╡ c71bb21f-678e-447e-917e-ac354411e2a1
grad(MarginalLikelihood(), md), grad(MarginalLikelihood(), hpmin, md)

# ╔═╡ 79fe56c1-1d70-4381-8236-f32a52d263f7
xp = reshape(collect(0:0.01:1), (1, :))

# ╔═╡ 320c10b9-b53e-4be4-8d59-4dc3d078f7b4
update_params!(md, hpmin)

# ╔═╡ e50be0b8-43cc-4cf4-87cd-e7af7e8047e3
yp, Σp = posterior(md, xp);

# ╔═╡ 40ca4cf4-2472-4eea-a713-63d613a8b0b1
yt = f(xp)

# ╔═╡ 87c7a9a5-4498-4d6c-8ce2-9919d77dcfef
yt .- yp

# ╔═╡ 17384162-ac84-4e71-ad0d-71b78e279451
let
	p = plot()
	xx = dropdims(x, dims=1)
	xpx = dropdims(xp, dims=1)
	scatter!(p, xx, y)
	yerr = diag(Σp)
	plot!(p, xpx, yp, ribbon=yerr)
end

# ╔═╡ Cell order:
# ╠═a09e1670-45de-11ec-0a4b-9d1ced6b0199
# ╠═72620701-24dd-460e-8b49-0c53dcdd577f
# ╠═a0895e98-2d01-48ae-b249-22655a21b71c
# ╠═a786390d-fc31-4eed-9d97-1bf29485e2e6
# ╠═13c44ff7-c021-4ae7-a5b2-3f6a49a864c5
# ╠═68f21ff2-f6d8-4132-ac1e-978d9be87d94
# ╠═9bc25e84-35d5-4e7f-8c26-81ecdfb12b7a
# ╠═deb35813-a714-4aed-8ef3-72d008f591f1
# ╠═823cb765-acb8-439c-8197-44f3cb9ebd41
# ╠═ce861ed4-f98a-4052-bebe-590c15cc259b
# ╠═c71bb21f-678e-447e-917e-ac354411e2a1
# ╠═79fe56c1-1d70-4381-8236-f32a52d263f7
# ╠═320c10b9-b53e-4be4-8d59-4dc3d078f7b4
# ╠═e50be0b8-43cc-4cf4-87cd-e7af7e8047e3
# ╠═40ca4cf4-2472-4eea-a713-63d613a8b0b1
# ╠═87c7a9a5-4498-4d6c-8ce2-9919d77dcfef
# ╠═17384162-ac84-4e71-ad0d-71b78e279451
