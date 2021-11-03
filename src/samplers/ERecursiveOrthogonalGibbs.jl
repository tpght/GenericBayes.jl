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
function step(rng, outer_model::BayesModel, sampler::ERecursiveOrthogonalGibbs,
              current_state=nothing; kwargs...) where T<:Real
    # Check if l divides p
    if (dimension(outer_model) % sampler.l != 0)
        error("RecursiveOrthogonalGibbs: l does not divide model dimension")
    end

    # First, generate an initial state if required
    if (current_state == nothing)
        state = max_posterior(outer_model, zeros(dimension(outer_model)))
        return state, state
    end

    """
        OrthogonalGibbs

    Performs the OrthogonalGibbs method by sampling on a k-dimensional m-flat then
    e-flat submanifold
    """
    function OrthogonalGibbs(θ0::Vector{<:Real}, geometry::Bregman, model::BayesModel)
        # Check if we're on the last e-flat submanifold
        if(length(θ0) <= sampler.l)
            # Sample on the remaining l variables
            samples = AbstractMCMC.sample(rng, model, sampler.subsampler,
                                          sampler.subsamples, progress=false)
            return samples[end]
        end

        # If sampler.l is small, then k is potentially large.
        k = length(θ0) - sampler.l

        # First, sample on the m-flat submanifold defined by first length(θ0) - l dual
        # co-ordinates being fixed. This is a l dimensional submanifold.
        η0 = legendre_dual(θ0, geometry, model, k)[1:k]
        mconditional_model = MFlatConditionalGibbs(model, geometry, η0, θ0[1:k])

        # Sample from density defined by mflat_log_target
        samples = AbstractMCMC.sample(rng,
                                      mconditional_model,
                                      sampler.subsampler, sampler.subsamples, progress=false)

        # samples[end] is now the l = length(θ0) - k resampled primal components.
        θ1 = inverse_legendre_dual([η0; samples[end]], geometry, model, k, x0 = θ0[1:k])

        # Use recursion to sample the remaining k variables.
        econditional_model = EFlatConditionalGibbs(model, θ1[(k+1):end])
        inherited = EFlatGibbs(geometry, θ1[(k+1):end])
        sample = OrthogonalGibbs(θ1[1:k], inherited, econditional_model)

        return [sample; samples[end]]
    end

    # Call the orthogonal gibbs function
    state = OrthogonalGibbs(current_state, sampler.geometry, outer_model)

    # Return new state
    return state, state
end
