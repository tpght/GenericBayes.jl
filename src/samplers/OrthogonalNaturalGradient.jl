import AbstractMCMC.AbstractSampler, AbstractMCMC.step
export OrthogonalNaturalGradient

"""
    OrthogonalNaturalGradient

Orthogonal Gibbs, recursing on e-flat submanifolds.
"""
struct OrthogonalNaturalGradient <: AbstractSampler
    geometry::Bregman           # Geometry to be used in the sampler
    subsampler::AbstractSampler # Sampler to be used on each m-geodesic
    subsamples::Int             # Number of times to run the embedded sampler
end


"""
    step(rng, model, sampler,
              state=nothing; kwargs...)

One iteration of the orthogonal natural gradient method.
"""
function step(rng, outer_model::BayesModel, sampler::OrthogonalNaturalGradient,
              current_state=nothing; kwargs...) where T<:Real

    # First, generate an initial state if required
    if (current_state == nothing)
        # NOTE: Can't use the mode for this, because natural gradient is zero at
        # the mode.
        state = max_posterior(outer_model, zeros(dimension(outer_model))) + rand(Normal(), dimension(outer_model))
        return state, state

    end
    """
        ONG

    Performs the ONG method by sampling on a k-dimensional m-flat then
    e-flat submanifold
    """
    function ONG(θ0::Vector{<:Real}, geometry::Bregman, model::BayesModel)
        # Get dimension of model
        p = length(θ0)

        # Check if we're on the last m-geodesic
        # TODO Can also stop before we get to this point.
        if(p == 1)
            # Sample on the remaining l variables (e-flat)
            samples = AbstractMCMC.sample(rng, model, sampler.subsampler,
                   sampler.subsamples, progress=false)

            return samples[end]
        end

        # Compute the gradient with respect to primal co-ordinates.
        log_density(x) = log_posterior_density(model, x)
        gradient = ForwardDiff.gradient(log_density, θ0)

        # Convert current point to dual co-ordinates (take gradient of generator)
        η0 = legendre_dual(θ0, geometry, model)

        # Parameterize the dual-geodesic to sample along
        dg(t) = η0 + t .* gradient

        # Sample from the dual geodesic η0 + t * gradient
        function mgeodesic_log_target(t)
            primal = inverse_legendre_dual(dg(t), generator, x0=θ0)
            log_density(primal) - logabsdetmetric(primal, generator)
        end

        # Sample from density defined by mflat_log_target
        samples = AbstractMCMC.sample(rng,
                                      LogDensityModel(mgeodesic_log_target, 1),
                                      sampler.subsampler, sampler.subsamples, progress=false)

        # samples[end] is now the resampled parameter along the dual geodesic.
        # Convert to a primal co-ordinate.
        @show samples[end]
        θ1 = inverse_legendre_dual(dg(samples[end]), generator, x0 = θ0)

        # Construct a basis for the space orthogonal to gradient
        B = (diagm(ones(p)) - (1.0 / norm(gradient, 2)^2) * (gradient * gradient'))[:, 1:(p-1)]

        # Use recursion to sample the remaining k variables.
        restricted_log_density(x) = log_density(θ1 + B * x)
        restricted_generator(x) = generator(θ1 + B * x)
        sample = ONG(restricted_log_density, zeros(p-1), restricted_generator)

        return θ1 + B * sample
    end

    # Call the orthogonal gibbs function
    state = ONG(current_state, sampler.geometry, outer_model)

    # Return new state
    return state, state
end
