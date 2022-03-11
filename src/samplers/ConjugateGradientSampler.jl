import AbstractMCMC.AbstractSampler, AbstractMCMC.step
export ConjugateGradientSampler

"""
    ConjugateGradientSampler

Conjugate gradient sampler (Algorithm 8 in the thesis). Uses the posterior
Bregman geometry.
"""
struct ConjugateGradientSampler{T<:Real} <: AbstractSampler
    initial_θ::Vector{T}
    stop::Int                   # Stop after this many steps
end


"""
    step(rng, model::????, sampler::ConjugateGradientSampler,
              state=nothing; kwargs...)

p iterations of the conjugate gradient sampler.
"""
function step(rng, model::GaussianInverse{T}, sampler::ConjugateGradientSampler{T},
              current_state=nothing; kwargs...) where T<:Real
    p = dimension(model)

    # First, generate an initial state if required
    if (current_state == nothing)
        return sampler.initial_θ, sampler.initial_θ
    end

    g = model.Λ * current_state - model.w
    v = -copy(g)
    r = -copy(g)
    s = zeros(p)

    stop = min(p, sampler.stop)

    for i in 1:stop
        ρ = v' * model.Λ * v

        λ = (rand(Normal()) / sqrt(ρ)) - ((v' * g) / ρ)

        # Compute update to current position
        dx = λ .* v

        # Update sample
        current_state = current_state .+ dx

        # Check for any loss of conjugacy
        # NOTE: Is this the best place to check?
        if(i == stop)
            return current_state, current_state
        end

        if(r' * r ≈ 0.0)
            @warn "Conjugate gradient sampler exited early after $i iterations"
            return current_state, current_state
        end

        # TODO: This can be rewritten in terms of previous g and Λ * v
        g = model.Λ * current_state .- model.w               # Gradient (negative residual)

        δ = (g' * r) / (r' * r)     # Component of x in r direction
        s = s .+ (δ .* r)             # Euclidean projection into current Krylov space
        r = s .- g                   # Compute next Euclidean-orthogonal direction
        v = r .- ((r' * g) / (ρ * λ)) .* v # Compute next conjugate direction
    end

    return current_state, current_state
end