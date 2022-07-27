# Sampling a Gaussian process

```example samp
using GaussianProcessRegression
using CairoMakie

n = 100
x = reshape(collect(range(0,1,n)), (1,:))
gp = GaussianProcess(x -> zero(eltype(x)), SquaredExp())
y = sample(gp, x)
lines(x, y)
save("sampl.svg", current_figure())
```

![]("sampl.svg")