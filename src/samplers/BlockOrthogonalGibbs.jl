import AbstractMCMC.AbstractSampler, AbstractMCMC.step
export BlockOrthogonalGibbs

"""
    BlockOrthogonalGibbs

Orthogonal Gibbs sampling algorithm: generalizes e/m-recursive versions
"""
struct BlockOrthogonalGibbs <: AbstractSampler
    geometry::Bregman           # Geometry to be used in the sampler
    block_size::Int                      # Dimension of the e-flat submanifold
    subsampler::AbstractSampler # Sampler to be used on each e-flat submanifold
    subsamples::Int                  # Number of times to run the embedded sampler
end


"""
    step(rng, model::BayesModel, sampler::BlockOrthogonalGibbs,
              state=nothing; kwargs...)

One iteration of the orthogonal gibbs method.
"""
function step(rng, model::BayesModel, sampler::BlockOrthogonalGibbs,
              current_state=nothing; kwargs...) where T<:Real

    p = dimension(model)
    k = sampler.block_size

    # Check if l divides p
    @assert p % k == 0 "BlockOrthogonalGibbs: k does not divide model dimension"

    # First, generate an initial state if required
    if (current_state == nothing)
        θ = max_posterior(model, zeros(p))
        # η = legendre_dual(θ, sampler.geometry, model, p-k)
        # return θ, [θ, η]
        return θ, θ
    end

    # θ = copy(current_state[1])
    # η = copy(current_state[2])

    # η1 = legendre_dual(θ, sampler.geometry, model, p - k)
    # @show norm(η - η1, 2)

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

            # NOTE Here we evaluate a big sub-matrix.
            # what if we simply evaluate the k × k block on the diagonal, in
            # the correct position for the block?
            logπ(model, embed) - logabsdetmetric(embed, sampler.geometry,
                                                 model, k * (block - 1))
        end

        # Draw samples from the k-dimensional dist. with log-density eflat_target
        subsamples = AbstractMCMC.sample(rng,
                                         LogDensityModel(target, k),
                                         sampler.subsampler, sampler.subsamples,
                                         progress=false)

        # Save a subsample in the current block
        θ[block_inds] .= subsamples[end]


        # embed into the ambient space
        embed = inverse_legendre_dual([ηc; θ[[block_inds; upper_inds]]],
                                      sampler.geometry,
                                      model, k * (block - 1),
                                      x0=θ[lower_inds])

        # NOTE: Why do we have to save these indices?
        θ[[lower_inds; block_inds]] .= embed[[lower_inds; block_inds]]

        # Compute dual co-ordinates for this block
        # NOTE: η[lower_inds] should be equal to [lower_inds] of dual coordinate
        # of embed = ηc, so only have to save this block.
        # NOTE: This is unnecessary at the final step...
        if(block < nblocks)
            η[block_inds] .= ForwardDiff.gradient(x -> bregman_generator(
                                             [embed[lower_inds]; x; embed[upper_inds]],
                sampler.geometry, model),
                                                  embed[block_inds])
        else
            η[block_inds] = θ[block_inds]
        end
    end

    # The first returned value is the sample, i.e. in primal co-ordinates
    # The second value is the "state"; for this sampler, that's the pair of
    # primal and dual variables.
    return θ, θ
end
