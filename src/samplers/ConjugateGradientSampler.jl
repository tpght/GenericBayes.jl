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
    Constructor; sets stop to dimension of initial_θ
"""
function ConjugateGradientSampler(initial_θ::Vector{T}) where T<:Real
    ConjugateGradientSampler(initial_θ, length(initial_θ))
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
    # TODO Do these have to be copied?
    v = -copy(g)
    r = -copy(g)
    # s = zeros(p)

    stop = min(p, sampler.stop)

    for i in 1:stop
        # Compute matrix-vector product
        q = model.Λ * v

        ρ = v' * q
        if(ρ < eps())
            @warn "Conjugate gradient sampler exited early after $i iterations;
                    ρ = $ρ"
            return current_state, current_state
        end

        λ = (rand(rng, Normal()) / sqrt(ρ)) - ((v' * g) / ρ)

        # Compute update to current position
        dx = λ .* v

        # Update sample
        current_state = current_state .+ dx

        # Check for any loss of conjugacy
        # NOTE: Is this the best place to check?
        if(i == stop)
            return current_state, current_state
        end

        if(r' * r < eps())
            @warn "Conjugate gradient sampler exited early after $i iterations;
                    r' * r = $(r' * r)"
            return current_state, current_state
        end

        # TODO: This can be rewritten in terms of previous g and Λ * v
        g = g .+ (λ .* q)

        δ = (g' * r) / (r' * r)     # Component of x in r direction
        # s = s .+ (δ .* r)             # Euclidean projection into current Krylov space
        # r = s .- g                   # Compute next Euclidean-orthogonal direction
        r = (1 + δ) .* r .- λ .* q
        v = r .- ((r' * g) / (ρ * λ)) .* v # Compute next conjugate direction
    end

    return current_state, current_state
end
