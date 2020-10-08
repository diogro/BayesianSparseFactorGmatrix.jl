	function sampleBSF_G!(Posterior::PosteriorSample, genetic_effects::GeneticEffects, 
	                      Factors::LatentFactors, resid::Residuals, 
	                      interaction_effects::InteractionEffects, fixed_effects::FixedEffects,
	                      D::InputData, Pr::Priors)
	    #precalculate some matrices
	    #invert the random effect covariance matrices
	    Ainv = inv(D.A)
	    A_2_inv = Matrix{Int}(I, D.n, D.n); #Z_2 random effects are assumed to have covariance proportional to the identity. Can be modified.

	    #pre-calculate transformation parameters to diagonalize aI + bZAZ for fast
	    #inversion: inv(aI + bZAZ) = 1/b*u*diag(1./(s+a/b))*u'
	    #uses singular value decomposition of ZAZ for stability when ZAZ is low
	    #rank()
	    # eig_ZAZ.vectors = u; -> eig_ZAZ.U
	    # eig_ZAZ.values = diag(s); -> eig_ZAZ.S

	    ZAZ = D.Z_1' * D.A * D.Z_1;
	    eig_ZAZ = svd(ZAZ);

	    Design=[D.X; D.Z_1];
	    Design2 = Design * Design';
	    svd_Design_Ainv = makeSVDdict(Array(BlockDiagonal([fixed_effects.cov, Ainv])), Design2);
	    Qt_Design = svd_Design_Ainv["Q"]' * Design;

	    #fixed effects + random effects 1
	    #diagonalize mixed model equations for fast inversion: 
	    #inv(a*blkdiag(fixed_effects.cov,Ainv) + b*[X Z_1]"[X Z_1]'') = Q*diag(1./(a.*s1+b.*s2))*Q"
	    #inv(Array(BlockDiagonal([fixed_effects.cov, Ainv])) + Design2) ≈ svd_Design_Ainv["Q"] * Diagonal(1. ./ (svd_Design_Ainv["s1"]+svd_Design_Ainv["s2"])) * svd_Design_Ainv["Q"]'


	    #random effects 2
	    #as above; but for random effects 2. Here; fixed effects will be conditioned on; not sampled simultaneously. Otherwise identical.
	    if(D.r2 > 1)
	        Design_Z2 = Z_2
	        Design2_Z2 = Design_Z2*Design_Z2'
	        svd_Z2_2_A2inv = makeSVDdict(A_2_inv, Design2_Z2)
	        Qt_Z2 = svd_Z2_2_A2inv["Q"]'*Design_Z2
	    end

	    #genetic effect variances of factor traits
	    # diagonalizing a*Z_1*Z_1' + b*Ainv for fast inversion
	    #diagonalize mixed model equations for fast inversion: 
	    # inv(a*Z_1*Z_1" + b*Ainv) = Q*diag(1./(a.*s1+b.*s2))*Q'
	    #similar to fixed effects + random effects 1 above; but no fixed effects.
	    ZZt = D.Z_1 * D.Z_1'
	    svd_ZZ_Ainv = makeSVDdict(ZZt, Array(Ainv))
	    # inv(Array(ZZt) + Array(Ainv)) ≈ svd_ZZ_Ainv["Q"] * Diagonal(1. ./ (svd_ZZ_Ainv["s1"]+svd_ZZ_Ainv["s2"])) * svd_ZZ_Ainv["Q"]'

	    sp_num=0
	    @showprogress 1 "Running Gibbs sampler..." for i = 1:Pr.nrun

	        ##fill in missing phenotypes
	        ##conditioning on everything else()
	        #phenMissing = isnan(Y_full)
	        #if sum(sum(phenMissing))>0
	        #    meanTraits = fixed_effects.B*X +  genetic_effects.d*Z_1 ...
	        #        + interaction_effects.W*Z_2 + Factors.Lambda*Factors.scores
	        #    meanTraits = meanTraits';        
	        #    resids = bsxfun[@times,randn(size(Y_full)),1./sqrt(resid.ps')]
	        #    Y[phenMissing] = meanTraits[phenMissing] + resids[phenMissing]
	        #end

	        #sample Lambda
	        #conditioning on W; X; F; marginalizing over D
	        Ytil = D.Y' - fixed_effects.B * D.X - interaction_effects.W * D.Z_2
	        sample_lambda!(Factors, Ytil, resid, genetic_effects, eig_ZAZ)

	        #sample fixed effects + random effects 1 [[B D]']
	        #conditioning on W; F; L
	        Ytil = D.Y' - interaction_effects.W * D.Z_2 - Factors.Lambda*Factors.scores
	        N = genetic_effects.n + fixed_effects.b
	        location_sample = sample_means(Ytil, Qt_Design, N, resid, genetic_effects.ps, svd_Design_Ainv)
	        fixed_effects.B = location_sample[:,1:fixed_effects.b]
	        genetic_effects.d = location_sample[:, fixed_effects.b+1 : fixed_effects.b+genetic_effects.n ]

	        #sample random effects 2
	        #conditioning on B; D; F; L
	        N = interaction_effects.n
	        if N>0
	            Ytil = D.Y'-fixed_effects.B * D.X - genetic_effects.d * D.Z_1 - Factors.Lambda * Factors.scores
	            location_sample = sample_means(Ytil, Qt_Z2, N, resid, interaction_effects.ps, svd_Z2_2_A2inv)
	            interaction_effects.W = location_sample
	        end

	        #sample factor h2
	        #conditioning on F; marginalizing over U
	        sample_h2s_discrete!(Factors, eig_ZAZ)

	        #sample genetic effects [U]
	        #conditioning on F; Factor h2
	        sample_Us!(Factors, genetic_effects, svd_ZZ_Ainv, D.Z_1)

	        #sample F
	        #conditioning on U; Lambda; B; D; W; factor h2s
	        Ytil = D.Y' - fixed_effects.B * D.X - genetic_effects.d * D.Z_1 - interaction_effects.W * D.Z_2
	        sample_factors_scores!(Ytil, Factors, resid, genetic_effects, D.Z_1)


	        # -- Update ps -- #
	        Lambda2 = Factors.Lambda .^ 2 
	        as_p = Factors.df/2. + 0.5
	        bs_p = 2. ./ (Factors.df .+ broadcast(*, Lambda2, Factors.tauh'))
	        for i = 1:D.p, j = 1:Factors.k
	            Factors.psijh[i, j] = rand(Gamma(as_p, bs_p[i, j]))
	        end

	        #continue from previous Y residual above
	        Ytil = Ytil - Factors.Lambda * Factors.scores
	        inv_bs = 1. ./ (resid.bs .+ 0.5 * sum(Ytil .^ 2, dims=2)) #model residual precision
	        for i = 1:D.p
	            resid.ps[i] = rand(Gamma(resid.as + 0.5 * D.n, inv_bs[i]))
	        end

	        #random effect 1 [D] residual precision
	        inv_b_g = 1. ./ (genetic_effects.bs .+ 0.5 * sum(genetic_effects.d .^ 2, dims=2))
	        for i = 1:D.p
	         genetic_effects.ps[i] = rand(Gamma(genetic_effects.as + 0.5 * genetic_effects.n, inv_b_g[i]))
	        end

	        #n = interaction_effects.n
	        #interaction_effects.ps=gamrnd[interaction_effects.as + 0.5*n,1./(interaction_effects.bs+0.5*sum(interaction_effects.W.^2,dims=2))]; #random effect 2 [W] residual precision

	        #------Update delta & tauh------#
	        sample_delta!(Factors, Lambda2)

	        #---update precision parameters----#
	        Factors.Plam = broadcast(*, Factors.psijh, Factors.tauh')

	        # ----- adapt number of factors to samples ----#
	        update_k!( Factors, genetic_effects, Pr.b0, Pr.b1 ,i , Pr.epsilon, Pr.prop, D.Z_1 )

	        # -- save sampled values [after thinning] -- #
	        if i%Pr.thin==0 && i > Pr.burn
	            sp_num = Int64((i-Pr.burn)/Pr.thin)
	            save_posterior_samples!(sp_num, Pr, D, Posterior, resid, fixed_effects,
	                genetic_effects, Factors, interaction_effects)
	        end
	    end
	end