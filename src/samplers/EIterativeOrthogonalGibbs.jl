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
function step(rng, model::BayesModel, sampler::EIterativeOrthogonalGibbs,
              current_state=nothing; kwargs...) where T<:Real

    p = dimension(model)
    l = sampler.l

    # Check if l divides p
    if (p % l != 0)
        error("RecursiveOrthogonalGibbs: l does not divide model dimension")
    end

    # First, generate an initial state if required
    if (current_state == nothing)
        state = max_posterior(model, zeros(p))
        return state, state
    end

    # θ contains the current state
    # NOTE is it necessary to copy here?
    θ = copy(current_state)

    # Compute dual co-ordinates for everything but the final block
    η = legendre_dual(θ, sampler.geometry, model, p - l)

    # Compute number of blocks
    nblocks = Int(p / l)

    for block = nblocks:(-1):1
        # CONDITIONED ON dual components: [1 : (block - 1) * k]
        # CONDITIONED ON primal components: [block * k + 1 :end ]
        upper_inds = (block*l+1):p
        lower_inds = 1:((block-1)*l)
        block_inds = ((block-1)*l+1):(block*l)
        θc = θ[upper_inds]
        ηc = η[lower_inds] # If we're on the first block, this is empty

        # Sample block conditionally on values of η filled in thus far, and
        # values of θ from the next block onwards.
        function target(x)
            # Evaluate joint density of mixed co-ordinates
            θp = [x; θc]

            # embed into the ambient space
            embed = inverse_legendre_dual([ηc; θp],
                                          sampler.geometry,
                                          model, l * (block - 1))

            logπ(model, embed) - logabsdetmetric(embed, sampler.geometry,
                                                 model, l * (block - 1))
        end

        # Draw samples from the k-dimensional dist. with log-density eflat_target
        subsamples = AbstractMCMC.sample(rng,
                                         LogDensityModel(target, l),
                                         sampler.subsampler, sampler.subsamples,
                                         progress=false)

        # Save a subsample in the current block
        θ[block_inds] .= subsamples[end]

        # embed into the ambient space
        embed = inverse_legendre_dual([ηc; θ[[block_inds; upper_inds]]],
                                      sampler.geometry,
                                      model, l * (block - 1))

        # NOTE: Why do we have to save these indices?
        θ[[lower_inds; block_inds]] .= embed[[lower_inds; block_inds]]

        # Compute dual co-ordinates for this block
        η[block_inds] = ForwardDiff.gradient(x -> bregman_generator(
                                                 [embed[lower_inds]; x; embed[upper_inds]],
                                                 sampler.geometry, model), embed[block_inds])
    end

    return θ, θ
end
