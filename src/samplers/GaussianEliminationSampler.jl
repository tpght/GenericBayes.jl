import AbstractMCMC.AbstractSampler, AbstractMCMC.step
export GaussianEliminationSampler

"""
    GaussianEliminationSampler

Conjugate gradient sampler (Algorithm 8 in the thesis). Uses the posterior
Bregman geometry.
"""
struct GaussianEliminationSampler{T<:Real} <: AbstractSampler
    initial_θ::Vector{T}
end

"""
    step(rng, model::????, sampler::GaussianEliminationSampler,
              state=nothing; kwargs...)

p iterations of the conjugate gradient sampler.
"""
function step(rng, model::GaussianInverse{T}, sampler::GaussianEliminationSampler{T},
              current_state=nothing; kwargs...) where T<:Real
    p = dimension(model)

    # First, generate an initial state if required
    if (current_state == nothing)
        return sampler.initial_θ, sampler.initial_θ
    end


    # Preallocate variables for algorithm
    # TODO: some of these may be unnecessary.
    μ = zeros(p)
    w = copy(model.w)
    d = zeros(p)
    v = diagm(ones(p))
    A = Matrix(model.Λ)

    for i in 1:p
        d[i] = 1.0 / A[i,i]               # Equivalent of ρ in CG sampler

        z = rand(rng, Normal()) * sqrt(d[i])   # Sample random component of conjugate update
        e = (w[i] - A[i,:]' * current_state) * d[i]   # Deterministic component

        # Update state
        dx = (z + e) * v[:,i]
        current_state = current_state + dx

        μ = μ + d[i] * w[i] * v[:,i]

        # Gaussian elimination to get next conjugate vector
        for j in (i+1):p
            C = A[j,i] / A[i,i]

            # Take C * row i from row j
            A[j,:] = A[j,:] - C * A[i,:]

            # Conjugate vectors are the rows resulting from
            # Gaussian elimination on the identity matrix
            v[:,j] = v[:,j] - C * v[:,i]

            # Gaussian elimination on RHS of equation
            w[j] = w[j] - C * w[i]
        end
    end

    return current_state, current_state
end
