import AbstractMCMC.AbstractSampler, AbstractMCMC.step
export OrthogonalGibbs

function log_cond_density(α, θ, η, j, model, geometry)
    ξ = [η[1:(j-1)]; α; θ[(j+1):end]]
    if(j == 1)
        return log_posterior_density(model,ξ), ξ
    end

    θ = inverse_legendre_dual(ξ, geometry, model, j-1; x0 = θ[1:(j-1)])
    log_dens = log_posterior_density(model, θ) -
        logabsdetmetric(θ, geometry, model, j-1)
    return log_dens, θ
end

"""
    OrthogonalGibbs

Orthogonal Gibbs sampler with blocksize=1.
"""
struct OrthogonalGibbs{T<:Real} <: AbstractSampler
    initial_θ::Vector{T}        # Initial point at which to start sampler
    geometry::Bregman           # Geometry to be used in the sampler
    w::T                        # Initial window size for slice subsampling
    m::Int                      # Maximum number of times to stepout (slice sampler)
    nsubsamples::Int            # Number of times to run the embedded sampler
    return_last::Bool           # If true, final iterate of algorithm is returned as a sample
end

function OrthogonalGibbs(initial_θ::Vector{T}, geometry::Bregman, w::T,
                         m::Int, nsubsamples::Int) where T<:Real
    return_last=true
    OrthogonalGibbs(initial_θ, geometry, w,
                    m, nsubsamples, return_last)
end

"""
    step(rng, model::BayesModel, sampler::ProductManifoldHMC,
              state=nothing; kwargs...)

One iteration of the e-recursive orthogonal gibbs method.
"""
function step(rng, model::BayesModel, sampler::OrthogonalGibbs,
              current_state=nothing; kwargs...) where T<:Real
    # Get dimension of model
    p = dimension(model)

    # First, generate an initial state if required
    if (current_state == nothing)
        θ = copy(sampler.initial_θ)

        # Compute gradient of Bregman generator
        η = legendre_dual(θ, sampler.geometry, model)

        return θ, (p, θ, η)
    end

    # Extract last iteration index and last sample from state
    last_j = current_state[1]
    θ = copy(current_state[2])
    η = current_state[3]

    # Restart iterations if required.
    if (last_j >= p) last_j=0 end

    for j = (last_j+1):p
        # Resample the j^th component conditionally on 1,..,(j-1) dual variables,
        # (j+1),..,p primal
        α = θ[j]

        # Evaluate log density of the conditional.
        logf = log_posterior_density(model, θ) -
               logabsdetmetric(θ, sampler.geometry, model, (j-1))

        for i=1:sampler.nsubsamples
            # TODO pinv can be simplified; A has orthogonal columns
            x0 = θ[1:(j-1)]

            # Firstly sample y ~ Unif(0, f(α))
            # Equivalently to sampling z = g(α) - e
            # where g = log(f) and e ~ Exponential(1).
            z = logf - rand(rng, Exponential(1.0))

            # Slice is now defined by S = {x: z < g(x)}

            # Stepping out procedure. (Fig 3 in Neal)
            # TODO move to separate function
            U = rand(rng, Uniform())
            L = α - sampler.w * U
            R = L + sampler.w
            V = rand(rng, Uniform())
            J = floor(sampler.m * V)
            K = (sampler.m-1)-J

            while (J >0 && log_cond_density(L, θ, η, j, model, sampler.geometry)[1] > z)
                L = L -sampler.w
                J = J-1
            end

            while (K >0 && log_cond_density(R, θ, η, j, model, sampler.geometry)[1] > z)
                R = R +sampler.w
                K = K-1
            end

            # Shrinkage and sampling procedure.
            Lb = L
            Rb = R

            while (true)
                U = rand(rng, Uniform())
                α1 = Lb + U *(Rb - Lb)
                logfα1, θ_α1 = log_cond_density(α1, θ, η, j, model, sampler.geometry)
                if(z < logfα1)
                    # Accept!
                    α = α1
                    logf = logfα1
                    θ = θ_α1
                    break
                end

                if(α1 < α)
                    Lb = α1
                else
                    Rb = α1
                end

                if(Lb ≈ Rb)
                    @error "Interval shrunk to 0"
                end
            end

        end

        # Compute gradient of Bregman generator
        η = legendre_dual(θ, sampler.geometry, model)
        last_j = j

        if(sampler.return_last == false) break end
    end

    return θ, (last_j, θ, η)
end
