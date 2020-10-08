
function save_posterior_samples!(sp_num::Int64, Pr::Priors, D::InputData, Posterior::PosteriorSample, 
                                 resid::Residuals, fixed_effects::FixedEffects, 
                                 genetic_effects::GeneticEffects, Factors::LatentFactors, 
                                 interaction_effects::InteractionEffects)
    #save posteriors. Re-scale samples back to original variances.
    sp = Pr.sp;
    VY = D.Var_Y';       
    Lambda = broadcast(*, Factors.Lambda, sqrt.(VY'));     #re-scale by Y variances
    G_h2 = Factors.h2;
    U = genetic_effects.U;     
    delta = Factors.delta;
    genetic_ps = genetic_effects.ps ./ VY';
    resid_ps = resid.ps ./ VY';

    # save factor samples
    Lambda = Lambda[:,1:Factors.k];

    Posterior.Lambda[sp_num] = copy(Lambda);
    Posterior.delta[sp_num] = copy(delta);
    Posterior.G_h2[sp_num] = copy(G_h2);
    Posterior.U[sp_num] = copy(U);

    Posterior.no_f[sp_num] = Factors.k;

    Posterior.ps[:,sp_num] = copy(genetic_ps);
    Posterior.resid_ps[:,sp_num] = copy(resid_ps);

    # save B,U,W
    Posterior.B = Posterior.B + (broadcast(*, fixed_effects.B, sqrt.(VY'))  ./ sp);
    Posterior.d = Posterior.d + (broadcast(*, genetic_effects.d, sqrt.(VY')) ./ sp);
    Posterior.W = Posterior.W + (broadcast(*, interaction_effects.W, sqrt.(VY')) ./ sp); 

end

	function PosteriorMeans(Posterior::PosteriorSample, D::InputData, Pr::Priors)
    kmax = convert(Int64, maximum(Posterior.no_f));

    Lambda_est = zeros(D.p, kmax);
    G_s = zeros(D.p, D.p, Pr.sp);
    P_s = zeros(D.p, D.p, Pr.sp);
    E_s = zeros(D.p, D.p, Pr.sp);
    G_est = zeros(D.p, D.p);
    P_est = zeros(D.p, D.p);
    E_est = zeros(D.p, D.p);

    factor_h2s_est = zeros(kmax);
    for j=1:Pr.sp
        Lj  = Posterior.Lambda[j];
        h2j = Posterior.G_h2[j];

        Pj = Lj * Lj'                   + diagm(1. ./ (Posterior.ps[:,j])) + diagm(1. ./(Posterior.resid_ps[:,j]));
        Gj = Lj * diagm(     h2j) * Lj' + diagm(1. ./ (Posterior.ps[:,j]));
        Ej = Lj * diagm(1 .- h2j) * Lj'                                    + diagm(1. ./(Posterior.resid_ps[:,j]));


        P_s[:, :, j] = copy(Pj);
        G_s[:, :, j] = copy(Gj);
        E_s[:, :, j] = copy(Ej);

        P_est = P_est + Pj./Pr.sp;
        G_est = G_est + Gj./Pr.sp;
        E_est = E_est + Ej./Pr.sp;

        k_local = size(h2j, 1)
        Lambda_est[:, 1:k_local]  = Lambda_est[:, 1:k_local] + Lj./Pr.sp;
        factor_h2s_est[1:k_local] = factor_h2s_est[1:k_local] + h2j./Pr.sp;

    end

    posterior_mean = Dict("G" => G_est,
      "P" => P_est,
      "E" => E_est,
      "Gs" => G_s,
      "Ps" => P_s,
      "Es" => E_s,
      "Lambda" => Lambda_est,
      "F_h2s" => factor_h2s_est)
    posterior_mean
end