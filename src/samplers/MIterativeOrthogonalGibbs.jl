import AbstractMCMC.AbstractSampler, AbstractMCMC.step
export MIterativeOrthogonalGibbs

"""
    ERecursiveOrthogonalGibbs

Orthogonal Gibbs, recursing on e-flat submanifolds.
"""
struct MIterativeOrthogonalGibbs <: AbstractSampler
    geometry::Bregman           # Geometry to be used in the sampler
    k::Int                      # Dimension of the e-flat submanifold
    subsampler::AbstractSampler # Sampler to be used on each e-flat submanifold
    subsamples::Int                  # Number of times to run the embedded sampler
end


"""
    step(rng, model::BayesModel, sampler::ProductManifoldHMC,
              state=nothing; kwargs...)

One iteration of the e-recursive orthogonal gibbs method.
"""
function step(rng, model::BayesModel, sampler::MIterativeOrthogonalGibbs,
              current_state=nothing; kwargs...) where T<:Real

    p = dimension(model)
    k = sampler.k

    # Check if l divides p
    @assert p % k == 0 "MIterativeOrthogonalGibbs: k does not divide model dimension"

    # First, generate an initial state if required
    if (current_state == nothing)
        state = ones(p)
        return state, state
    end

    # θ contains the current state
    # NOTE is it necessary to copy here?
    θ = copy(current_state)

    # Compute dual co-ordinates for everything but the final block
    # TODO Save this between Markov steps
    η = legendre_dual(θ, sampler.geometry, model, p - k)

    # Compute number of blocks
    nblocks = Int(p / k)

    for block = 1:nblocks
        # CONDITIONED ON dual components: [1 : (block - 1) * k]
        # CONDITIONED ON primal components: [block * k + 1 :end ]
        upper_inds = (block*k+1):p
        lower_inds = 1:((block-1)*k)
        block_inds = ((block-1)*k+1):(block*k)
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
                                          model, k * (block - 1),
                                          x0=θ[lower_inds])

            logπ(model, embed) - logabsdetmetric(embed, sampler.geometry,
                                                 model, k * (block - 1))
        end

        # Draw samples from the k-dimensional dist. with log-density eflat_target
        sampler.subsampler.initial_θ .= θ[block_inds]
        subsamples = AbstractMCMC.sample(rng,
                                        LogDensityModel(target, k),
                                        sampler.subsampler, sampler.subsamples,
                                        progress=false)

        # Save a subsample in the current block
        θ[block_inds] .= subsamples.value[end, :].data # A convoluted way of
        # getting the last sample

        # embed into the ambient space
        embed = inverse_legendre_dual([ηc; θ[[block_inds; upper_inds]]],
                                      sampler.geometry,
                                      model, k * (block - 1),
                                      x0=θ[lower_inds])

        # NOTE: Why do we have to save these indices?
        θ[[lower_inds; block_inds]] .= embed[[lower_inds; block_inds]]

        # Compute dual co-ordinates for this block
        η[block_inds] = ForwardDiff.gradient(x -> bregman_generator(
                                                 [embed[lower_inds]; x; embed[upper_inds]],
                                                 sampler.geometry, model), embed[block_inds])

    end

    return θ, θ
end
