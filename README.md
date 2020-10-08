# BayesianSparseFactorGmatrix.jl

Julia implementation of the BSF-G algorithm from Runcie &amp; Mukherjee (2013)

# Instalation 

In Julia v1.5:

```
Pkg.add(PackageSpec(url="https://github.com/diogro/BayesianSparseFactorGmatrix.jl"))
```

# Basic Usage

```
using Distributions
using LinearAlgebra
using Kronecker
using BayesianSparseFactorGmatrix

function randomData(n, p, b)
    X = ones(b, n)             # fixed effect matrix
    X[2, :] = randn(n);
    X[3, :] = rand([0,1], n);

    beta = reshape([1., 1, 1, 1,
                    1,  1, 1, 1,
                    2,  2, 2, 2], (p, b))

    # beta = reshape([1., 2, 3, 
    #                 1,  1, 1], (p, b))'

    A = rand(Normal(0.4, 0.01), n, n)
    A = (A + A') ./ 2
    A[diagind(A)] .= 1

    G = reshape([1., 0.8, 0, 0,
                 0.8,  1, 0, 0,
                 0,    0, 1, 0.4,
                 0,    0, 0.4, 1], (p, p))'

    ad = reshape(Array(cholesky(Array(kronecker(A, G))).L) * randn(p * n), (p, n))'

    Z_1 = Matrix{Float64}(I, n, n) # pedigree model matrix
    Z_2 = zeros(0,n);          # second random effect


    d = Normal(1, sqrt(0.5))
    Y = (beta * X)' + Z_1 * ad + rand(d, n, p)
    [Y, X, A, Z_1, ad]
end
Y1, X1, A1, Z_11, ad = randomData(100, 4, 3);

posterior_mean, Posterior, D, Pr, fixed_effects = runBSFGModel(Y1, X1, A1, Z_11);

posterior_mean["G"]
```
