cholcov = function(x::Array{Float64, 2})
    tol = 1e-6
    if(isposdef(x))
        out = Array(cholesky(x).U)
    else
        eigX = eigen(Hermitian(x))
        val = sqrt.(eigX.values[eigX.values .> tol])
        vec = eigX.vectors[:,eigX.values .> tol]
        out = Diagonal(val) * vec'
    end
    out
end

function makeSVDdict(A::Array{Float64, 2}, B::Array{Float64, 2})
    U,V,Q,C,S,R = svd(cholcov(A), cholcov(B))
    #svd_Design_Ainv.Q = inv(q)'
    #svd_Design_Ainv. 
    #svd_Design_Ainv.s2 = diag(S2'*S2)
    #Qt_Design = svd_Design_Ainv.Q'*Design;    
    H = R * Q'
    q = H'
    Q = ("Q", inv(q)')
    s1 = ("s1", Array(diag(C' * C)))
    s2 = ("s2", Array(diag(S' * S)))
    Dict([Q, s1, s2])
end

function cov2cor(x::Array{Float64, 2})
    sds = sqrt.(diag(x))
    x ./ (sds * sds')
end