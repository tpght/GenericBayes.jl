import AbstractMCMC.AbstractSampler, AbstractMCMC.step
export MRescursiveOrthogonalGibbs

"""
    MRescursiveOrthogonalGibbs

Orthogonal Gibbs, recursing on e-flat submanifolds.
"""
struct MRescursiveOrthogonalGibbs <: AbstractSampler
    geometry::Bregman           # Geometry to be used in the sampler
    k::Int                      # Dimension of the e-flat submanifold
    subsampler::AbstractSampler # Sampler to be used on each m-flat submanifold
    subsamples::Int                  # Number of times to run the embedded sampler
end


"""
    step(rng, model::BayesModel, sampler::ProductManifoldHMC,
              state=nothing; kwargs...)

One iteration of the e-recursive orthogonal gibbs method.
"""
function step(rng, model::BayesModel, sampler::MRescursiveOrthogonalGibbs,
              current_state=nothing; kwargs...) where T<:Real
    # Check if k divides p
    if (dimension(model) % sampler.k != 0)
        error("RecursiveOrthogonalGibbs: l does not divide model dimension")
    end

    # First, generate an initial state if required
    if (current_state == nothing)
        state = max_posterior(model, zeros(dimension(model)))
        return state, state
    end

    """
        OrthogonalGibbs

    Performs the OrthogonalGibbs method by sampling on a k-dimensional e-flat then
    m-flat submanifold
    """
    function OrthogonalGibbs(log_density::Function, θ0::Vector{<:Real}, generator::Function)
        # Check if we're on the last m-flat submanifold
        if(length(θ0) <= sampler.k)
            # Sample on the remaining l variables
            samples = AbstractMCMC.sample(rng, LogDensityModel(log_density, sampler.l), sampler.subsampler,
                   sampler.subsamples, progress=false)

            return samples[end]
        end

        l = length(θ0) - sampler.k

        # First, sample on the e-flat submanifold defined by last l = p - k
        # primal components being fixed. This is just a regular Gibbs update on
        # a k-dimensional hyperplane.
        restricted_log_density(x) = log_density([x; θ0[(k+1):end]])
        samples = AbstractMCMC.sample(rng,
                                      LogDensityModel(restricted_log_density, sampler.k),
                                      sample.subsampler, sampler.subsamples, progress=false)

        # Take the final sample
        θ1 = [samples[end]; θ0[(k+1):end]]

        # Next, sample on the m-flat submanifold defined by first k dual
        # co-ordinates being fixed. This is a (p-k)-dimensional m-flat submanifold.

        # Evaluate the k dual components.
        η1 = legendre_dual(θ1, generator, k)[1:k]

        function mflat_log_target(x)
            primal = inverse_legendre_dual([η1; x], generator, k, x0=θ1[1:k])
            log_density(primal) - logabsdetmetric(primal, generator, k)
        end

        function restricted_generator(x)
            primal = inverse_legendre_dual([η1; x], generator, k, x0=θ1[1:k])
            generator([primal; x])
        end

        # TODO what is the restricted generator on the m-flat manifold,
        # parameterized by remaining primal components θ?
        # By Lemma 3.5.1, should just be restriction of generator to
        # TODO Need to define metric, geometry on the manifold.
        restricted_generator(x) = generator([x; samples[end]]) -
        sample = OrthogonalGibbs(mflat_log_target, θ1[(k+1):end], restricted_generator)

        return [sample; samples[end]]
    end

    log_density(x) = log_posterior_density(model, x)
    generator(x) = bregman_generator(x, sampler.geometry, model)

    # Call the orthogonal gibbs function
    state = OrthogonalGibbs(log_density, current_state, generator)

    # Return new state
    return state, state
end
