function sample_lambda!(Factors::LatentFactors, Ytil::Array{Float64, 2}, 
    resid::Residuals, genetic_effects::GeneticEffects, eig_ZAZ::SVD)
    #Sample factor loadings (Factors.Lambda) while marginalizing over residual
    #genetic effects: Y - Z_2W = FL' + E, vec(E)~N(0,kron(Psi_E,In) + kron(Psi_U, ZAZ^T))
    # note: conditioning on W, but marginalizing over U.
    #  sampling is done separately by trait because each column of Lambda is
    #  independent in the conditional posterior
    # note: eig_ZAZ has parameters that diagonalize aI + bZAZ for fast
    #  inversion: inv(aI + bZAZ) = 1/b*Ur*diag(1./(eta+a/b))*Ur'
    p=resid.p
    k=Factors.k

    Ur = eig_ZAZ.U;
    eta = eig_ZAZ.S;
    FtU = Factors.scores*Ur;
    UtY = Ur' * Ytil';

    Zlams = rand(Normal(0,1),k,p);
    for j = 1:p
       FUDi  = genetic_effects.ps[j] * broadcast(*, FtU, 1. ./ (eta' .+ genetic_effects.ps[j]/resid.ps[j]));
       means = FUDi * UtY[:,j];
       Qlam  = FUDi*FtU' + diagm(Factors.Plam[j,:]); 
       Llam  = cholesky(Hermitian(Qlam)).L
       vlam  = Llam  \ means; 
       mlam  = Llam' \ vlam; 
       ylam  = Llam' \ Zlams[:,j];
       Factors.Lambda[j,:] = (ylam + mlam);
    end
end

function sample_means(Ytil::Array{Float64, 2}, Qt_Design::Array{Float64, 2}, N::Int64, 
  resid::Residuals, random_precision::Array{Float64, 1}, svd_Design_Ainv::Dict)
    # when used to sample [B;D]:
    # Y - FL' - Z_2W = XB + ZD + E, vec(E)~N(0,kron(Psi_E,In)). 
    # Note: conditioning on F, L and W.
    #  The vector [b_j;d_j] is sampled simultaneously. Each trait is sampled separately because their
    #  conditional posteriors factor into independent MVNs.
    # Note:svd_Design_Ainv has parameters to diagonalize mixed model equations for fast inversion: 
    #  inv(a*blkdiag(fixed_effects.cov,Ainv) + b*[X; Z_1][X; Z_1]') = Q*diag(1./(a.*s1+b.*s2))*Q'
    # Qt_Design = Q'*Design, which doesn't change each iteration. Design = [X;Z_1]
    #
    # function also used to sample W:
    #  Y - FL' - XB - ZD = Z_2W + E, vec(E) ~ N(0,kron(Psi_E,In)). 
    #  Here, conditioning is on B and D.

    p=resid.p;

    Q = svd_Design_Ainv["Q"];
    s1 = svd_Design_Ainv["s1"];
    s2 = svd_Design_Ainv["s2"];

    means = broadcast(*, Qt_Design * Ytil', resid.ps');
    location_sample = zeros(N,p);
    Zlams = randn(N,p);
    for j = 1:p,
        d = s1 * random_precision[j] + s2*resid.ps[j];
        mlam = broadcast(*, means[:,j], 1. ./ d);
        location_sample[:,j] = Q * (mlam + broadcast(*, Zlams[:,j], 1. ./ sqrt.(d)));
    end
    location_sample=location_sample';
    location_sample
end

function sample_h2s_discrete!(Factors::LatentFactors, eig_ZAZ::SVD)
    # sample factor heritibilties from a discrete set on [0,1)
    # prior places 50% of the weight at h2=0
    # samples conditional on F, marginalizes over U.

    Ur = eig_ZAZ.U;
    eta = eig_ZAZ.S;

    r = Factors.r;
    k = Factors.k;
    s = Factors.h2_divisions;

    log_ps = zeros(k,s);
    std_scores_b = Factors.scores*Ur;
    for i=1:s
        h2 = (i-1)/s;
        std_scores = Factors.scores;
        if h2 > 0
            std_scores = 1/sqrt(h2) * broadcast(*, std_scores_b, 1. ./ sqrt.(eta .+ (1-h2)/h2)');
            det = sum(log.((eta .+ (1-h2)/h2) * h2) / 2.);
        else       
            det = 0;
        end
        log_ps[:,i] = sum(logpdf(Normal(0, 1), std_scores), dims = 2) .- det; #Prior on h2
        if i==1
            log_ps = log_ps .+ log(s-1);
        end
    end
    for j=1:k
        norm_factor = max.(log_ps[j,:]) .+ log(sum(exp.(log_ps[j,:] - max.(log_ps[j,:]))));
        ps_j = exp.(log_ps[j,:] - norm_factor);
        log_ps[j,:] = ps_j;
        Factors.h2[j] = sum(rand() .> cumsum(ps_j)) / s;
    end
end

function sample_Us!(Factors::LatentFactors, genetic_effects::GeneticEffects, 
                    svd_ZZ_Ainv::Dict, Z_1::Array{Float64, 2})
    #samples genetic effects (U) conditional on the factor scores F:
    # F_i = U_i + E_i, E_i~N(0,s2*(h2*ZAZ + (1-h2)*I)) for each latent trait i
    # U_i = zeros(r,1) if h2_i = 0
    # it is assumed that s2 = 1 because this scaling factor is absorbed in
    # Lambda
    # svd_ZZ_Ainv has parameters to diagonalize a*Z_1*Z_1' + b*I for fast
    # inversion:

    Q = svd_ZZ_Ainv["Q"];
    s1 = svd_ZZ_Ainv["s1"];
    s2 = svd_ZZ_Ainv["s2"];

    k = Factors.k;
    n = genetic_effects.n;
    tau_e = 1. ./ (1 .- Factors.h2);
    tau_u = 1. ./ Factors.h2;
    b = Q' * Z_1 * broadcast(*, Factors.scores, tau_e)';
    z = randn(n,k);
    for j=1:k
        if tau_e[j]==1
            genetic_effects.U[j,:] = zeros(1,n);
        elseif isinf(tau_e[j])
            genetic_effects.U[j,:] = Factors.scores[j,:];
        else
            d = s2 * tau_u[j] + s1 * tau_e[j];
            mlam = broadcast(*, b[:,j], 1. ./ d);
            genetic_effects.U[j,:] = (Q*(mlam + broadcast(*, z[:,j], 1. ./ sqrt.(d))))';
        end
    end
end

function sample_factors_scores!(Ytil::Array{Float64, 2}, Factors::LatentFactors, 
    resid::Residuals, genetic_effects::GeneticEffects, Z_1::Array{Float64, 2})
#Sample factor scores given factor loadings, U, factor heritabilities and
#phenotype residuals

    k = Factors.k;
    n = Factors.n;
    Lambda = Factors.Lambda;
    Lmsg = broadcast(*, Lambda, resid.ps);
    tau_e = reshape(1. ./ (1. .- Factors.h2), k);
    S = cholesky(Hermitian(Lambda' * Lmsg + diagm(tau_e))).L;
    Meta = S' \ (S \ (Lmsg' * Ytil + broadcast(*, genetic_effects.U * Z_1 , tau_e)));
    Factors.scores = Meta + S' \ randn(k,n);   
end


function sample_delta!( Factors::LatentFactors, Lambda2_resid )
    #sample delta and tauh parameters that control the magnitudes of higher
    #index factor loadings.

    ad1 = Factors.ad1;
    ad2 = Factors.ad2;
    bd1 = Factors.bd1;
    bd2 = Factors.bd2;
    k = Factors.k;
    psijh = Factors.psijh;

    mat = broadcast(*, psijh, Lambda2_resid);
    n_genes = size(mat,1);
    ad = ad1 + 0.5*n_genes*k; 
    bd = bd1 + 0.5*(1/Factors.delta[1]) * sum(Factors.tauh .* sum(mat, dims = 1)');
    Factors.delta[1] = rand(Gamma(ad,1. / bd));
    Factors.tauh = cumprod(Factors.delta);

    for h = 2:k
        ad = ad2 + 0.5*n_genes*(k-h+1); 
        bd = bd2 + 0.5*(1. / Factors.delta[h])*sum(Factors.tauh[h:end].*sum(mat[:,h:end], dims = 1)');
        Factors.delta[h] = rand(Gamma(ad,1/bd));
        Factors.tauh = cumprod(Factors.delta);
    end
end

function  update_k!(Factors::LatentFactors, genetic_effects::GeneticEffects, 
    b0::Float64, b1::Float64, i::Int64, epsilon::Float64, prop::Float64, Z_1::Array{Float64, 2} )
#adapt the number of factors by dropping factors with only small loadings
#if they exist, or adding new factors sampled from the prior if all factors
#appear important. The rate of adaptation decreases through the chain,
#controlled by b0 and b1

    df = Factors.df;
    ad2 = Factors.ad2;
    bd2 = Factors.bd2;
    p = Factors.p;
    k = Factors.k;
    r = Factors.r;
    gene_rows = 1:p;
    Lambda = Factors.Lambda;

    # probability of adapting
    prob = 1/exp(b0 + b1*i);                
    uu = rand();

    # proportion of elements in each column less than eps in magnitude
    lind = mean(abs.(Lambda[1:p,:]) .< epsilon, dims = 1);    
    vec = lind .>= prop; num = sum(vec);       # number of redundant columns

    Factors.num = num;
    Factors.no_f[i] = k-num;

    if uu < prob && i>200
        if  i > 20 && num == 0 && all(lind .< 0.995) && k < 2*p 
            #add a column
            k=k+1;
            Factors.k = k;
            Factors.psijh = [Factors.psijh rand(Gamma(df/2, 2/df), p, 1)]
            Factors.delta = [Factors.delta; rand(Gamma(ad2, 1/bd2))];
            Factors.tauh = cumprod(Factors.delta);
            Factors.Plam = broadcast(*, Factors.psijh, Factors.tauh');
            Factors.Lambda = [Factors.Lambda randn(p,1) .* sqrt.(1 ./ Factors.Plam[:, k])];
            Factors.h2 = [Factors.h2; rand()];
            genetic_effects.U = [genetic_effects.U; randn(1,genetic_effects.n)];
            Factors.scores = [Factors.scores; genetic_effects.U[k,:]' * Z_1 + randn(1,r) .* sqrt(1-Factors.h2[k])];
        elseif num > 0      # drop redundant columns
            nonred = Array(1:k)[vec[:] .== 0]; # non-redundant loadings columns
            k = max(k - num, 1);
            Factors.k = k;
            Factors.Lambda = Lambda[:,nonred];
            Factors.psijh = Factors.psijh[:,nonred];
            Factors.scores = Factors.scores[nonred,:];
            for red = setdiff(1:k-1,nonred)
                #combine deltas so that the shrinkage of kept columns doesnt
                #decrease after dropping redundant columns
                Factors.delta[red+1] = Factors.delta[red+1] * Factors.delta[red];
            end
            Factors.delta = Factors.delta[nonred];
            Factors.tauh = cumprod(Factors.delta);
            Factors.Plam = broadcast(*, Factors.psijh, Factors.tauh');
            Factors.h2 = Factors.h2[nonred];
            genetic_effects.U = genetic_effects.U[nonred,:];
        end
    end
    Factors.nofout[i+1]=k;
end