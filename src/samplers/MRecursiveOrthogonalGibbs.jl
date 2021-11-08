import AbstractMCMC.AbstractSampler, AbstractMCMC.step
export MRecursiveOrthogonalGibbs

"""
    MRescursiveOrthogonalGibbs

Orthogonal Gibbs, recursing on e-flat submanifolds.
"""
struct MRecursiveOrthogonalGibbs <: AbstractSampler
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
function step(rng, outer_model::BayesModel, sampler::MRecursiveOrthogonalGibbs,
              current_state=nothing; kwargs...) where T<:Real

    # k is a constant over each level of recursion
    # The dimension of the e-flat submanifold
    k = sampler.k

    # Check if k divides p
    if (dimension(outer_model) % k != 0)
        error("RecursiveOrthogonalGibbs: k does not divide model dimension")
    end

    # First, generate an initial state if required
    if (current_state == nothing)
        state = zeros(dimension(outer_model))
        return state, state
    end

    """
        OrthogonalGibbs

    Performs the OrthogonalGibbs method by sampling on a k-dimensional e-flat then
    m-flat submanifold
    """
    function OrthogonalGibbs(θ0::Vector{<:Real}, geometry::Bregman,
    model::BayesModel)
        # Check if we're on the last m-flat submanifold
        if(length(θ0) <= k)
            # Sample on the remaining variables
            samples = AbstractMCMC.sample(rng, model, sampler.subsampler,
                                          sampler.subsamples, progress=false)
            return samples[end]
        end

        # First, sample on the e-flat submanifold defined by last l = p - k
        # primal components being fixed. This is just a regular Gibbs update on
        # a k-dimensional hyperplane.
        econditional_model = EFlatConditionalGibbs(model, θ0[(k+1):end])
        samples = AbstractMCMC.sample(rng,
                                      econditional_model,
                                      sampler.subsampler, sampler.subsamples,
                                      progress=false)

        # Take the final sample, embed into total space
        θ1 = [samples[end]; θ0[(k+1):end]]

        # Next, sample on the m-flat submanifold defined by first k dual
        # co-ordinates being fixed. This is a (p-k)-dimensional m-flat submanifold.

        # Evaluate the k dual components.
        η1 = legendre_dual(θ1, geometry, model, k)[1:k]

        # Define the inherited geometry on the m-flat submanifold
        inherited = MFlatGibbs(geometry, η1)

        # function mflat_log_target(x)
        #     primal = inverse_legendre_dual([η1; x], geometry, model, k, x0=θ1[1:k])
        #     logπ(model, primal) - logabsdetmetric(primal, geometry, model, k)
        # end
        # mflat_model = LogDensityModel(mflat_log_target, length(θ0) - k)
        mconditional_model = MFlatConditionalGibbs(model, geometry, η1, θ1[1:k])

        θ_c = OrthogonalGibbs(θ1[(k+1):end], inherited, mconditional_model)

        # Embed into total space
        embed = inverse_legendre_dual([η1; θ_c], geometry, model, k, x0=θ1[1:k])

        return embed
    end

    # Call the orthogonal gibbs function
    state = OrthogonalGibbs(current_state, sampler.geometry, outer_model)

    # Return new state
    return state, state
end
