module BayesianSparseFactorGmatrix

	using Statistics 
	using Random
	using Distributions
	using LinearAlgebra
	using BlockDiagonals
	using Kronecker
	using ProgressMeter

	include("utils.jl")
	include("types.jl")
	include("sampling_functions.jl")
	include("posterior_utils.jl")
	include("BSFG.jl")

	## Initialize run parameters & prior hyper-parameters
	# run parameters 
	# _burn = 1000; 
	# _sp = 1000; 
	# _thin = 10

	# _b0 = 1.; 
	# _b1 = 0.0005

	# _epsilon = 1e-2
	# _h2_divisions = 100

	# # prior hyperparamters

	# _k_init = 20; # initial number of factors to initialize
	# _as = 2.; _bs = 1. / 10; # inverse gamma hyperparameters for model residuals; as well as non-factor random effects
	# _df = 3; # degrees of freedom for t-distribution of ARD prior on factor loadings
	# _ad1 = 2.1; _bd1 = 1. /20; # inverse gamma hyperparamters for first factor shrinkage multiplier [/delta_1]
	# _ad2 = 3.; _bd2 = 1.; # inverse gamma hyperparamters for remaining factor shrinkage multiplier [/delta_i, i \in 2...k]


	# _k_min = 1e-1; # minimum factor loading size to report in running status
	# _prop = 1.00;  # proportion of redundant elements within columns necessary to drop column
	function runBSFGModel(Y, X, A, Z_1, burn::Int64=1000, sp::Int64=1000, thin::Int64=10, 
	                      b0::Float64=1., b1::Float64=0.0005, epsilon::Float64=1e-2, 
	                      h2_divisions::Int64=100, k_init::Int64=20, 
	                      as::Float64=2., bs::Float64=0.1, df::Float64=3., 
	                      ad1::Float64=2.1, bd1::Float64=1. / 20, 
	                      ad2::Float64=3., bd2::Float64=1., 
	                      k_min::Float64=0.1, prop::Float64=1.)
	    Pr = Priors(burn, sp, thin, b0, b1, epsilon, h2_divisions, k_init, 
	                        as, bs, df, ad1, bd1, ad2, bd2, k_min, prop)
	    D = InputData(Y, X, A, Z_1);
	    resid = Residuals(Pr.as, Pr.bs, D.Y, D.p, rand(Gamma(Pr.as, 1. / Pr.bs), D.p))
	    Factors = LatentFactors(D.r1, D.n, D.p, Pr.k_init, Pr.df, Pr.ad1, Pr.bd1, Pr.ad2, Pr.bd2, 
	                            Pr.h2_divisions, Pr.sp, Pr.nrun)
	    genetic_effects = GeneticEffects(D.n, Pr.as, Pr.bs, D.r1, D.p, Factors.k, Factors.h2)
	    interaction_effects = InteractionEffects(Pr.as, Pr.bs, D.p, D.r2)
	    fixed_effects = FixedEffects(D.b, D.p)             
	    Factors.scores = genetic_effects.U * D.Z_1 + broadcast(*, randn(Factors.k, D.n), sqrt.(1. .- Factors.h2))
	    Posterior = PosteriorSample(D.n, D.p, Pr.k_init, Pr.sp, D.r1, D.r2, D.b)
	    
	    sampleBSF_G!(Posterior::PosteriorSample, genetic_effects::GeneticEffects, Factors::LatentFactors,
	             resid::Residuals, interaction_effects::InteractionEffects, fixed_effects::FixedEffects, 
	             D::InputData, Pr::Priors)
	    
	    posterior_mean = PosteriorMeans(Posterior, D, Pr)
	    
	    [posterior_mean, Posterior, D, Pr, fixed_effects]
	end

	export runBSFGModel, cov2cor;

end
