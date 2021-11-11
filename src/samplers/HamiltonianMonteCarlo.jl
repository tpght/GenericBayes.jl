import AbstractMCMC.AbstractSampler, AbstractMCMC.sample
export HamiltonianMonteCarlo

"""
    HamiltonianMonteCarlo

Random walk, Gaussian proposal with a spherical covariance.
"""
struct HamiltonianMonteCarlo{T<:Real} <: AbstractSampler
    dense::Bool                 # If false, use a diagonal preconditioner
    n_adapts::Integer           # Number of adaptation samples
    initial_θ::Vector{T}
end

function set_initial(sampler::HamiltonianMonteCarlo{T}, v::Vector{T}) where
    T<:Real
    sampler.initial_θ[1:length(v)] .= v
end

"""
    sample(rng, model::BayesModel, sampler::HamiltonianMonteCarlo{G}; kwargs...)

Sample from model using Hamiltonian Monte Carlo (interface to AdvancedHMC).
"""
function sample(model::BayesModel, sampler::HamiltonianMonteCarlo, N::Integer;
                kwargs...)
    GenericBayes.sample(Random.GLOBAL_RNG, model, sampler, N; kwargs...)
end
function sample(rng::AbstractRNG, model::BayesModel,
                sampler::HamiltonianMonteCarlo, N::Integer;
                kwargs...)

    # The following is adapted from the minimal example on
    # https://github.com/TuringLang/AdvancedHMC.jl

    start=time()

    # Use mode of the distribution as starting point
    initial_θ = sampler.initial_θ[1:dimension(model)]

    # Define the target distribution
    ℓπ(θ) = log_posterior_density(model, θ)

    # Define a Hamiltonian system
    metric = DiagEuclideanMetric(dimension(model))
    if(sampler.dense == true)
        metric = DenseEuclideanMetric(dimension(model))
    end
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

    # Define a leapfrog solver, with initial step size chosen heuristically
    initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
    integrator = Leapfrog(initial_ϵ)

    progress = false
    if(haskey(kwargs, :progress))
        progress = kwargs[:progress]
    end

    # Define an HMC sampler, with the following components
    #   - multinomial sampling scheme,
    #   - generalised No-U-Turn criteria, and
    #   - windowed adaption for step-size and diagonal mass matrix
    proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    # Run the sampler to draw samples from the specified Gaussian, where
    #   - `samples` will store the samples
    #   - `stats` will store diagnostic statistics for each sample
    samples, stats = sample(hamiltonian, proposal, initial_θ, N, adaptor,
                            sampler.n_adapts;
                            progress=progress)

    stop=time()
    times = (start=start,stop=stop)
    return bundle_samples(samples, model, sampler, samples[end],
                          MCMCChains.Chains, stats=times)
end
