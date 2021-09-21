import AbstractMCMC.AbstractSampler, Base.show, AbstractMCMC.step
export SphericalRandomWalk
using Distributions

"""
    SphericalRandomWalk

Random walk, Gaussian proposal with a spherical covariance.
"""
struct SphericalRandomWalk{T<:Real, P<:Parameter{T}} <: AbstractSampler
    StepSize::T                 # Standard deviation of random walk proposal
end

"""
    step(rng, model::BayesModel, sampler::ProductManifoldHMC,
              state=nothing; kwargs...)

One iteration of the product manifold HMC method.
"""
function step(rng, model::BayesModel, sampler::SphericalRandomWalk{T, P},
              current_state=nothing; kwargs...) where P<:Parameter{T} where T<:Real
    # First, generate an initial state if required
    if (current_state == nothing)
        state = P(zeros(dimension(model)))
        return state.components, state
    end

    # Generate a proposal
    proposal = current_state.components +
        rand(rng, MvNormal(dimension(model), sampler.StepSize))

    # Accept / reject
    logp = logπ(model, P(proposal)) - logπ(model, current_state)

    if(logp > 0  || log(rand(rng)) < logp)
        # accept
        return proposal, P(proposal)
    end

    # reject
    return current_state.components, current_state
end
