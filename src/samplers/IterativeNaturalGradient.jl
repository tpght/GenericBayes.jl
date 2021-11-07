import AbstractMCMC.AbstractSampler, AbstractMCMC.step
export IterativeNaturalGradient

"""
    OrthogonalGibbs

Orthogonal Gibbs sampling algorithm: generalizes e/m-recursive versions
"""
struct IterativeNaturalGradient <: AbstractSampler
    geometry::Bregman           # Geometry to be used in the sampler
    subsampler::AbstractSampler # Sampler to be used on each e-flat submanifold
    subsamples::Int                  # Number of times to run the embedded sampler
end


"""
    step(rng, model::BayesModel, sampler::OrthogonalGibbs,
              state=nothing; kwargs...)

One iteration of the orthogonal gibbs method.
"""
function step(rng, model::BayesModel, sampler::IterativeNaturalGradient,
              current_state=nothing; kwargs...) where T<:Real

    p = dimension(model)

    # First, generate an initial state if required
    if (current_state == nothing)
        θ = max_posterior(model, zeros(p))
        # η = legendre_dual(θ, sampler.geometry, model, p-k)
        # return θ, [θ, η]
        return θ, θ
    end

    # θ contains the current state
    # NOTE is it necessary to copy here?
    θ = copy(current_state)

    # Compute dual co-ordinates for everything but the final block
    # TODO Save this between Markov steps
    η = legendre_dual(θ, sampler.geometry, model, p - 1)

    # Stores the basis (Gram-Schmidt-ed vectors)
    A = diagm(ones(p))

    log_density(x) = log_posterior_density(model, x)

    for block = 1:p
        # Compute the gradient at the current position
        # Note: could also be natural gradient.
        gradient = ForwardDiff.gradient(log_density, θ)
        Ap = A[:,1:(block-1)]
        ηc = Ap' * η

        # Gram-Schmidt with respect to Euclidean inner-product on A-basis
        # The following is a numerically stable Gram-Schmidt
        # Shamelessly taken from https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
        A[:, block]=gradient
        for j=1:(block-1)
            A[:,block]=A[:,block]-(A[:,j]'*A[:,block]
                       /(norm(A[:,j]))^2) * A[:,j]
        end
        A[:, block] = A[:, block]/norm(A[:, block]);

        # Sample block conditionally on values of η filled in thus far, and
        # values of θ from the next block onwards.
        function target(x)
            # Evaluate joint density of mixed co-ordinates

            # embed into the ambient space
            # NOTE: β component is where x goes, I think!
            # Project onto (Im A)^⟂, A = Ap here...
            # That gives b := B * β.
            # NOW the component of b in the direction of the ORTHOGONALIZED,
            # GRAM-SCHMIDTED gradient is what needs updating.
            b = θ - Ap * pinv(Ap) * θ
            b = b + (x[1] - (A[:,block]' * b)) * A[:,block]
            x0 = Ap' * θ
            embed = inverse_legendre_dual(ηc, b, Ap,
                                          sampler.geometry,
                                          model, x0=x0)

            # NOTE Here we evaluate a big sub-matrix.
            # what if we simply evaluate the k × k block on the diagonal, in
            # the correct position for the block?
            # A^T G A
            G = metric(embed, sampler.geometry, model)
            jac_term = logabsdet(Ap' * G * Ap)[1]
            logπ(model, embed) - jac_term
        end

        # Draw samples from the k-dimensional dist. with log-density eflat_target
        subsamples = AbstractMCMC.sample(rng,
                                         LogDensityModel(target, 1),
                                         sampler.subsampler, sampler.subsamples,
                                         progress=false)

        # Save a subsample in the current block
        # pinv(A)  = A' if A is ortho
        b = θ - Ap * pinv(Ap) * θ
        b = b + (subsamples[end][1] - A[:,block]' * b) * A[:,block]
        x0 = Ap' * θ

        embed = inverse_legendre_dual(ηc, b, Ap,
                                      sampler.geometry,
                                      model, x0=x0)

        # NOTE:  Why do we have to save these indices?
        Ap = A[:, 1:block]
        # @show length(θ)
        # @show length(embed)
        # @show size(Ap)
        θ .= θ + Ap * pinv(Ap) * (embed - θ)

        # Compute dual co-ordinates for this block
        # NOTE: η[lower_inds] should be equal to [lower_inds] of dual coordinate
        # of embed = ηc, so only have to save this block.
        # NOTE: This is unnecessary at the final step...
        # block_η = θ' * A[:,block]
        # if(block < nblocks)
        #     block_η .= ForwardDiff.gradient(x -> bregman_generator(
        #                                     [embed[lower_inds]; x; embed[upper_inds]],
        #                                     sampler.geometry, model),
        #                                     embed[block_inds])
        # end
        # # Update the relevant component of η
        # η .= η + (block_η - (η' * A[:,block])) * A[:,block]
        η .= legendre_dual(θ, sampler.geometry, model)
    end

    # The first returned value is the sample, i.e. in primal co-ordinates
    return θ, θ
end
