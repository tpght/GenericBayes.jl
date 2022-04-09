import AbstractMCMC.AbstractSampler, AbstractMCMC.step
export CholeskySampler

"""
    CholeskySampler

Conjugate gradient sampler (Algorithm 8 in the thesis). Uses the posterior
Bregman geometry.
"""
struct CholeskySampler <: AbstractSampler end

"""
    step(rng, model::????, sampler::CholeskySampler,
              state=nothing; kwargs...)

p iterations of the conjugate gradient sampler.
"""
function step(rng, model::GaussianInverse, sampler::CholeskySampler,
              current_state=nothing; kwargs...)
    p = dimension(model)

    z = rand(rng, Normal(), p)

    # Compute cholesky factor of Λ
    chol = cholesky(model.Λ)

    # Now, Λ = LL^T. Thus Λ^-1 = L^-T L^-1, and so L^-T z ~ N(0, Λ^-1)
    b = chol.L \ model.w
    x = chol.U \ (z + b)

    return x, x
end
