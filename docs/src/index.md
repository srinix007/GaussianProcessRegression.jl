# Gaussian Process Regression

```@docs
SquaredExp
```

```@example 1
using GaussianProcessRegression

x = rand(2,5)
cov = SquaredExp()
hp = rand(dim_hp(cov,2))
krn = kernel(cov, hp, x)
```

Covariance between different vecotors:

```@example 1
xp = rand(2, 7)
kxxp = kernel(cov, hp, x, xp)
```