# --- Initialize variables --- #
#residual parameters. This structure holds the priors hyperparamters for
#the gamma prior on the model residual variances. It also holds the current
#estimate of the residual precision

mutable struct Residuals
    as::Float64
    bs::Float64
    Y::Array{Float64,2}
    p::Int64
    ps::Array{Float64,1}
end

#Factors. This struct holds all information about the latent factors
#including the current number of factors; priors hyperparameters for the
#factor Loadings; as well as current values of the factor loadings; their
#precision; the factor scores; & the genetic heritability of each factor()

mutable struct LatentFactors
    r::Int64            
    n::Int64
    p::Int64
    k::Int64
    df::Int64
    ad1::Float64
    bd1::Float64
    ad2::Float64
    bd2::Float64
    h2_divisions::Int64        #discretizations of heritability
    sp::Int64
    nrun::Int64

    psijh::Array{Float64,2}    #individual loadings precisions
    delta::Array{Float64,1}    # components of tauh
    tauh::Array{Float64,1}     #extra shrinkage of each loading column
    Plam::Array{Float64,2}     #total precision of each loading
    Lambda::Array{Float64,2}   #factor loadings
    h2::Array{Float64,1}       #factor heritability
    
    num::Int64
    no_f::Array{Float64,1}
    nofout::Array{Float64,1}
    
    scores::Array{Float64, 2}
    
    function LatentFactors(r, n, p, k, df, ad1, bd1, ad2, bd2, h2_divisions, sp, nrun)
        psijh = rand(Gamma(df/2,2/df), p, k);
        delta = [rand(Gamma(ad1+10,1/bd1), 1); rand(Gamma(ad2,1/bd2), k-1)];
        tauh = cumprod(delta);
        Plam = broadcast(*, psijh, tauh')
        Lambda = randn(p,k) .* sqrt.(1. ./ Plam)
        h2 = rand(k)
        num = 0
        no_f=zeros(nrun)
        nofout = k*ones(nrun+1)
        scores = zeros(k, n)
        new(r, n, p, k, df, ad1, bd1, ad2, bd2, h2_divisions, sp, nrun, psijh, delta, tauh, Plam, Lambda, h2, num, no_f, nofout, scores)
    end
end

#genetic_effects. This structure holds information about latent genetic
#effects. U is latent genetic effects on factor traits. d is genetic
#effects on residuals of the factor traits. Plus prior hyperparameters for
#genetic effect precisions

mutable struct GeneticEffects
    n::Int64
    as::Float64
    bs::Float64
    ps::Array{Float64, 1}
    U::Array{Float64, 2}
    d::Array{Float64, 2}
    function GeneticEffects(n, as, bs, r, p, k, h2)
        ps = rand(Gamma(as,1/bs),p)
        U  = broadcast(*, randn(k,r), sqrt.(h2))
        d  = broadcast(*, randn(p,r), 1. ./ sqrt.(ps))
        new(n, as, bs, ps, U, d)
    end
end

#interaction_effects. Similar to genetic_effects structure except for
#additional random effects that do not contribute to variation in the
#factor traits

mutable struct InteractionEffects
    as::Float64
    bs::Float64
    ps::Array{Float64, 1}
    mean::Array{Float64, 2}
    n::Int64
    W::Array{Float64, 2}
    W_out::Array{Float64, 2}
    function InteractionEffects(as, bs, p, r2)
        ps = rand(Gamma(as, 1/bs), p)
        mean = zeros(p,r2)
        n = r2
        W = broadcast(*, randn(p,r2), 1. ./ sqrt.(ps))
        W_out = zeros(p,r2)
        new(as, bs, ps, mean, n, W, W_out)
    end
end

#fixed_effects hold B
mutable struct FixedEffects
    b::Int64                # number of fixed effects, including intercept
    cov::Array{Float64, 2}  #inverse covariance of fixed effects
    mean::Array{Float64, 2} #mean of fixed effects
    B::Array{Float64, 2}    #current estimate of fixed effects
    function FixedEffects(b, p)
        cov = zeros(b,b)
        mean = zeros(p,b)
        B = randn(p,b)
        new(b, cov, mean, B)
    end
end

#Posterior holds Posterior samples & Posterior means

mutable struct PosteriorSample
    Lambda::Array{Array{Float64,2},1}
    no_f::Array{Float64, 1}
    ps::Array{Float64, 2} 
    resid_ps::Array{Float64, 2} 
    B::Array{Float64, 2} 
    U::Array{Array{Float64,2},1} 
    d::Array{Float64, 2} 
    W::Array{Float64, 2} 
    delta::Array{Array{Float64,1},1}
    G_h2::Array{Array{Float64,1},1} 
    function PosteriorSample(n, p, k, sp, r, r2, b)
        Lambda = Array{Array{Float64,2},1}(undef,sp) 
        no_f = zeros(sp)
        ps = zeros(p, sp)
        resid_ps = zeros(p ,sp)
        B = zeros(p, b)
        U = Array{Array{Float64,2},1}(undef,sp) 
        d = zeros(p, r)
        W = zeros(p, r2)
        delta = Array{Array{Float64,1},1}(undef,sp) 
        G_h2 = Array{Array{Float64,1},1}(undef,sp) 
        new(Lambda, no_f, ps, resid_ps, B, U, d, W, delta, G_h2)
    end
end

struct InputData
    n::Int64
    p::Int64
    b::Int64
    r1::Int64
    r2::Int64
    Y_full::Array{Float64, 2}
    Y::Array{Float64, 2}
    Mean_Y::Array{Float64, 1}
    Var_Y::Array{Float64, 1}
    X::Array{Float64, 2}
    A::Array{Float64, 2}
    Z_1::Array{Float64, 2}
    Z_2::Array{Float64, 2}
    function InputData(Y, X, A, Z_1) 
        n = size(Y, 1)
        p = size(Y, 2)
        b = size(X, 1)
        r1 = size(Z_1, 1)
        r2 = 0
        Y_full = copy(Y);
        Mean_Y = zeros(p);
        Var_Y = zeros(p);
        for i in 1:p
            Mean_Y[i] = mean(Y[:, i]);
            Var_Y[i]  =  var(Y[:, i]);
        end
        Y = broadcast(-, Y, Mean_Y');                     
        Y = broadcast(*, Y, 1. ./ sqrt.(Var_Y)');  
        Z_2 = zeros(0,n);
        new(n, p, b, r1, r2, Y_full, Y, Mean_Y, Var_Y, X, A, Z_1, Z_2)
    end
    function InputData(Y, X, A, Z_1, Z_2) 
        n = size(Y, 1)
        p = size(Y, 2)
        b = size(X, 1)
        r1 = size(Z_1, 1)
        r2 = size(Z_2, 1)
        Y_full = copy(Y);
        Mean_Y = zeros(p);
        Var_Y = zeros(p);
        for i in 1:p
            Mean_Y[i] = mean(Y[:, i]);
            Var_Y[i]  =  var(Y[:, i]);
        end
        Y = broadcast(-, Y, Mean_Y');                     
        Y = broadcast(*, Y, 1. ./ sqrt.(Var_Y)');  
        new(n, p, b, r1, r2, Y_full, Y, Mean_Y, Var_Y, X, A, Z_1, Z_2)
    end
end


struct Priors
    burn::Int64
    sp::Int64
    thin::Int64
    
    b0::Float64
    b1::Float64
    
    epsilon::Float64
    h2_divisions::Int64
    
    k_init::Int64
    
    as::Float64
    bs::Float64
    
    df::Float64
    
    ad1::Float64
    bd1::Float64
    
    ad2::Float64
    bd2::Float64
    
    k_min::Float64
    prop::Float64
    
    nrun::Int64
    function Priors(burn, sp, thin, b0, b1, epsilon, h2_divisions, k_init, 
        as, bs, df, ad1, bd1, ad2, bd2, k_min, prop)
        nrun = burn+sp*thin
        new(burn, sp, thin, b0, b1, epsilon, h2_divisions, k_init, 
            as, bs, df, ad1, bd1, ad2, bd2, k_min, prop, nrun)
    end
end