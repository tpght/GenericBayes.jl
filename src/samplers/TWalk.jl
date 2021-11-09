import AbstractMCMC.AbstractSampler, AbstractMCMC.step
export TWalk

"""
    TWalk

Random walk, Gaussian proposal with a spherical covariance.
"""
struct TWalk{T<:Real} <: AbstractSampler
    initial_θ::Vector{T}
end


function sample(model::BayesModel, sampler::TWalk, N::Integer;
                kwargs...)
    GenericBayes.sample(Random.GLOBAL_RNG, model, sampler, N; kwargs...)
end
function sample(rng::AbstractRNG, model::BayesModel,
                sampler::TWalk, N::Integer;
                kwargs...)
    start=time()

    # Use mode of the distribution as starting point
    initial_θ = sampler.initial_θ
    initial_θp = initial_θ + rand(dimension(model))

    # Define the target distribution
    U(θ) = -log_posterior_density(model, θ)

    ###   The following initializes the TWalk object
    ###   The dimension of the parameter space is n
    Usupp(x) = true
    jtw = jtwalk( n=dimension(model), U=U, Supp=Usupp)

    #### This runs the T-Walk
    Run!(jtw, T=N-1, x0=initial_θ, xp0=initial_θp)

    samples = [sample[1:(end-1)] for sample in eachrow(jtw.Output)]

    stop=time()
    times = (start=start,stop=stop)

    return bundle_samples(samples, model, sampler, samples[end],
                          MCMCChains.Chains, stats=times)
end
