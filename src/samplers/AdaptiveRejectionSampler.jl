import AbstractMCMC.AbstractSampler, AbstractMCMC.sample
export AdaptiveRejectionSampler

"""
    AdaptiveRejectionSampler

Random walk, Gaussian proposal with a spherical covariance.
"""
struct AdaptiveRejectionSampler{T<:Real} <: AbstractSampler
    max_segments::Integer
    support::Tuple{T, T}
    search_range::Tuple{T, T}
end

AdaptiveRejectionSampler() = AdaptiveRejectionSampler(5, (-Inf, Inf), (-1000.0, 1000.0))

"""
    sample(rng, model::BayesModel, sampler::AdaptiveRejectionSampler{G}; kwargs...)

Sample from model using Hamiltonian Monte Carlo (interface to AdvancedHMC).
"""
function sample(model::BayesModel, sampler::AdaptiveRejectionSampler, N::Integer;
                kwargs...)
    GenericBayes.sample(Random.GLOBAL_RNG, model, sampler, N; kwargs...)
end
function sample(rng::AbstractRNG, model::BayesModel,
                sampler::AdaptiveRejectionSampler, N::Integer;
                kwargs...)
    # The following is adapted from:
    # https://github.com/mauriciogtec/AdaptiveRejectionSampling.jl
    # I had to modify this package; see CustomARS.jl

    mode = max_posterior(model, [rand()])[1]
    search_range = (mode - 0.01, mode + 0.01)

    # Define the target distribution
    ℓπ(θ) = log_posterior_density(model, [θ])
    ∇ℓπ(θ) = grad_log_posterior_density(model, [θ])[1]

    @show ∇ℓπ(search_range[1]), ∇ℓπ(search_range[2])
    @show ∇ℓπ(mode)

    rs = RejectionSampler(ℓπ, ∇ℓπ, sampler.support, search_range)
    samples = run_sampler!(rs, N);

    return [[sample] for sample in samples]
end
