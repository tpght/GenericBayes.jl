import AbstractMCMC.AbstractSampler, AbstractMCMC.step
export ERecursiveOrthogonalGibbs

"""
    ERecursiveOrthogonalGibbs

Orthogonal Gibbs, recursing on e-flat submanifolds.
"""
struct ERecursiveOrthogonalGibbs <: AbstractSampler
    geometry::Bregman           # Geometry to be used in the sampler
    l::Int                      # Dimension of the m-flat submanifold
    subsampler::AbstractSampler # Sampler to be used on each m-flat submanifold
    subsamples::Int                  # Number of times to run the embedded sampler
end


"""
    step(rng, model::BayesModel, sampler::ProductManifoldHMC,
              state=nothing; kwargs...)

One iteration of the e-recursive orthogonal gibbs method.
"""
function step(rng, model::BayesModel, sampler::ERecursiveOrthogonalGibbs,
              current_state=nothing; kwargs...) where T<:Real
    # Check if l divides p
    if (dimension(model) % sampler.l != 0)
        error("RecursiveOrthogonalGibbs: l does not divide model dimension")
    end

    # First, generate an initial state if required
    if (current_state == nothing)
        state = max_posterior(model, zeros(dimension(model)))
        return state, state
    end

    """
        OrthogonalGibbs

    Performs the OrthogonalGibbs method by sampling on a k-dimensional m-flat then
    e-flat submanifold
    """
    function OrthogonalGibbs(log_density::Function, θ0::Vector{<:Real}, generator::Function)
        # Check if we're on the last e-flat submanifold
        if(length(θ0) <= sampler.l)
            # Sample on the remaining l variables
            samples = AbstractMCMC.sample(rng, LogDensityModel(log_density, sampler.l), sampler.subsampler,
                   sampler.subsamples, progress=false)

            return samples[end]
        end

        k = length(θ0) - sampler.l

        # First, sample on the m-flat submanifold defined by first length(θ0) - l dual
        # co-ordinates being fixed. This is a l dimensional submanifold.
        η0 = legendre_dual(θ0, generator, k)[1:k]

        function mflat_log_target(x)
            primal = inverse_legendre_dual([η0; x], generator, k, x0=θ0[1:k])
            log_density(primal) - logabsdetmetric(primal, generator, k)
        end

        # Sample from density defined by mflat_log_target
        samples = AbstractMCMC.sample(rng,
                                      LogDensityModel(mflat_log_target, sampler.l),
                                      sampler.subsampler, sampler.subsamples, progress=false)

        # samples[end] is now the l = length(θ0) - k resampled primal components.
        θ1 = inverse_legendre_dual([η0; samples[end]], generator, k, x0 = θ0[1:k])

        # Use recursion to sample the remaining k variables.
        restricted_log_density(x) = log_density([x; samples[end]])
        restricted_generator(x) = generator([x; samples[end]])
        sample = OrthogonalGibbs(restricted_log_density, θ1[1:k], restricted_generator)

        return [sample; samples[end]]
    end

    log_density(x) = log_posterior_density(model, x)
    generator(x) = bregman_generator(x, sampler.geometry, model)

    # Call the orthogonal gibbs function
    state = OrthogonalGibbs(log_density, current_state, generator)

    # Return new state
    return state, state
end
