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
function step(rng, model::BayesModel, sampler::OrthogonalNaturalGradient,
              current_state=nothing; kwargs...) where T<:Real

    # First, generate an initial state if required
    if (current_state == nothing)
        # NOTE: Can't use the mode for this, because natural gradient is zero at
        # the mode.
        state = max_posterior(model, zeros(dimension(model))) + rand(Normal(), dimension(model))
        return state, state
    end

    """
        ONG
    Performs the ONG method by sampling on a k-dimensional m-flat then
    e-flat submanifold
    """
    function ONG(log_density::Function, θ0::Vector{<:Real}, generator::Function,
                 A::Matrix{T}) where T<:Real
        # Get dimension of model
        p = length(θ0)

        # Check if we're on the last m-geodesic
        # TODO Can also stop before we get to this point.
        if(p == 1)
            # Sample on the remaining l variables (e-flat)
            samples = AbstractMCMC.sample(rng, LogDensityModel(log_density, 1), sampler.subsampler,
                   sampler.subsamples, progress=false)

            return samples[end]
        end

        # Compute the gradient with respect to primal co-ordinates.
        gradient = ForwardDiff.gradient(log_density, θ0)

        # Convert current point to dual co-ordinates (take gradient of generator)
        η0 = legendre_dual(θ0, generator)

        # Orthogonalize A
        A = hcat(A, gradient)
        block=size(A)[2]
        for j=1:(block-1)
            A[:,block]=A[:,block]-(A[:,j]'*A[:,block]
                    /(norm(A[:,j]))^2) * A[:,j]
        end
        # This is the matrix B
        A[:, block] = A[:, block]/norm(A[:, block]);

        function log_target(x)
            # η0 can be decomposed as Aδ + Bβ
            # B ∈ R^(p × 1), A ∈ R^(p × (p-1))
            η1 = η0 + A[:,block] * (x - (A[:,block]' * η0))
            θ1 = inverse_legendre_dual(η1, generator, x0 = [gradient' * θ0])
            G = metric(θ1, generator)
            G = A[:, 1:(block-1)]' * G * A[:, 1:(block-1)]
            log_density(θ1) - logabsdet(G)[1]
        end

        # Sample from density defined by mflat_log_target
        samples = AbstractMCMC.sample(rng,
                                      LogDensityModel(log_target, 1),
                                      sampler.subsampler, sampler.subsamples, progress=false)

        η1 = η0 + A[:,block] * (x - (A[:,block]' * η0))
        θ1 = inverse_legendre_dual(η1, generator, x0 = [gradient' * θ0])

        # Construct a basis for the space orthogonal to gradient
        B = (diagm(ones(p)) - (1.0 / norm(gradient, 2)^2) * (gradient * gradient'))[:, 1:(p-1)]

        # Use recursion to sample the remaining k variables.
        @show θ1 + B * ones(length(θ1) - 1)
        restricted_log_density(x) = log_density(θ1 + B * x)
        restricted_generator(x) = generator(θ1 + B * x)
        sample = ONG(restricted_log_density, zeros(p-1), restricted_generator, A)

        return θ1 + B * sample
    end

    log_density(x) = log_posterior_density(model, x)
    generator(x) = bregman_generator(x, sampler.geometry, model)

    # Call the orthogonal gibbs function
    state = ONG(log_density, current_state, generator, zeros(0,0))

    # Return new state
    return state, state
end
