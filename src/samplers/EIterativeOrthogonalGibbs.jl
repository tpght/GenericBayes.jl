import AbstractMCMC.AbstractSampler, AbstractMCMC.step
export EIterativeOrthogonalGibbs

"""
    ERecursiveOrthogonalGibbs

Orthogonal Gibbs, recursing on e-flat submanifolds.
"""
struct EIterativeOrthogonalGibbs <: AbstractSampler
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
function step(rng, outer_model::BayesModel, sampler::EIterativeOrthogonalGibbs,
              current_state=nothing; kwargs...) where T<:Real

    p = dimension(outer_model)
    l = sampler.l

    # Check if l divides p
    if (p % l != 0)
        error("RecursiveOrthogonalGibbs: l does not divide model dimension")
    end

    # First, generate an initial state if required
    if (current_state == nothing)
        state = max_posterior(outer_model, zeros(p))
        return state, state
    end

    # Set initial point, geometry and model
    θ0 = current_state
    geometry = sampler.geometry
    model = outer_model

    # Create array to store sample
    full_sample = zeros(p)

    # Compute number of blocks
    nblocks = Int(length(current_state) / l)

    for block = 1:nblocks
        # Check if we're on the final submanifold
        # i.e. base case
        if(length(θ0) <= l)
            # Sample on the remaining l variables
            # These are the first l primal components
            # θ1,..,θl  - where l = p - k
            samples = AbstractMCMC.sample(rng, model, sampler.subsampler,
                                          sampler.subsamples, progress=false)
            # NOTE: This assumes last block is size l
            # as it should be, since l should divide p
            full_sample[1:l] = samples[end]
            return full_sample, full_sample
        end

        # NOTE: k changes at each iteration!!
        k = length(θ0) - l

        # First, sample on the m-flat submanifold defined by first length(θ0) - l dual
        # co-ordinates being fixed. This is a l dimensional submanifold.
        η0 = legendre_dual(θ0, geometry, model, k)[1:k]
        mconditional_model = MFlatConditionalGibbs(model, geometry, η0, θ0[1:k])

        # Sample from density defined by mflat_log_target
        samples = AbstractMCMC.sample(rng,
                                      mconditional_model,
                                      sampler.subsampler,
                                      sampler.subsamples,
                                      progress=false)

        # Add the resampled l components to the accumulator
        bblock = nblocks - block
        full_sample[(1 + l * bblock):(l*(bblock + 1))] = samples[end]

        # samples[end] is now the l = length(θ0) - k resampled primal components.
        θ1 = inverse_legendre_dual([η0; samples[end]], geometry, model, k, x0 = θ0[1:k])

        # Change the relevant parameters; equivalent to recursion
        model = EFlatConditionalGibbs(model, θ1[(k+1):end])
        geometry = EFlatGibbs(geometry, θ1[(k+1):end])
        θ0 = θ1[1:k]
    end
end
