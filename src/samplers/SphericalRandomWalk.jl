import AbstractMCMC.AbstractSampler, AbstractMCMC.step
export SphericalRandomWalk

"""
    SphericalRandomWalk

Random walk, Gaussian proposal with a spherical covariance.
"""
struct SphericalRandomWalk{T<:Real} <: AbstractSampler
    StepSize::T                 # Standard deviation of random walk proposal
end

"""
    step(rng, model::BayesModel, sampler::ProductManifoldHMC,
              state=nothing; kwargs...)

One iteration of the random walk metropolis method.
"""
function step(rng, model::BayesModel, sampler::SphericalRandomWalk{T},
              current_state=nothing; kwargs...) where T<:Real
    # First, generate an initial state if required
    if (current_state == nothing)
        state = max_posterior(model, zeros(dimension(model)))
        return state, state
    end

    # Generate a proposal
    proposal = current_state .+
        rand(rng, Normal(0.0, sampler.StepSize), dimension(model))

    # Accept / reject
    logp = logπ(model, proposal) - logπ(model, current_state)

    if(logp > 0  || log(rand(rng)) < logp)
        # accept
        return proposal, proposal
    end

    # reject
    return current_state, current_state
end