import AbstractMCMC.AbstractSampler, AbstractMCMC.step
export OrthogonalGradient

function log_cond_density(α, r, c, δ, model, Ap, geometry, x0)
    b = α .* r  + c

    if(Ap == Float64[] || size(Ap)[2] == 0)
        return log_posterior_density(model,b), b
    end

    θ = inverse_legendre_dual(δ[1:size(Ap)[2]], b, Ap, geometry, model; x0 = x0)
    log_dens = log_posterior_density(model, θ) - logabsdetmetric(θ, geometry, model, Ap)
    return log_dens, θ
end

"""
    OrthogonalGradient

Orthogonal Gradient sampler
"""
struct OrthogonalGradient{T<:Real} <: AbstractSampler
    initial_θ::Vector{T}        # Initial point at which to start sampler
    geometry::Bregman           # Geometry to be used in the sampler
    w::T                        # Initial window size for slice subsampling
    m::Int                      # Maximum number of times to stepout (slice sampler)
    nsubsamples::Int            # Number of times to run the embedded sampler
    stop::Int                   # Number of steps after which to restart the sampler.
end

function OrthogonalGradient(initial_θ::Vector{T}, geometry::Bregman,
                            w::T, m::Int, nsubsamples::Int) where T<:Real
    stop = length(initial_θ)
    OrthogonalGradient(initial_θ, geometry, w, m, nsubsamples, stop)
end

"""
    step(rng, model::BayesModel, sampler::ProductManifoldHMC,
              state=nothing; kwargs...)

One iteration of the e-recursive orthogonal gibbs method.
"""
function step(rng, model::BayesModel, sampler::OrthogonalGradient,
              current_state=nothing; kwargs...) where T<:Real
    # Get dimension of model
    p = dimension(model)

    # First, generate an initial state if required
    if (current_state == nothing)
        state = sampler.initial_θ[1:p]
        return state, state
    end

    # θ contains the current state
    # NOTE is it necessary to copy here?
    θ = copy(current_state)

    # Allocate space for variables
    A = diagm(ones(p))
    δ = zeros(p)
    c = copy(θ)

    # How many iterations should be done?
    stop = min(p, sampler.stop)

    # Compute gradient of Bregman generator
    g = legendre_dual(θ, sampler.geometry, model)

    for j = 1:stop
        # Gram-Schmidt step; orthogonalize
        r = A * δ - g

        c = c - ((r' * c)/ (r' * r)) .* r

        α0 = (r' * θ) / (r' * r)

        # (Do subsampling here)
        # (Re-use embedding, store in θ)
        Ap = A[:, 1:(j-1)]

        # Compute scale for random walk
        prec = norm(r, 2)
        if(prec == 0.0)
            @warn "WARNING: r' * r = 0"
            return θ, θ
        end

        # Now sample from the relevant conditional (see thesis)
        # For this we use slice sampling (see Neal's paper).
        α = α0

        # Evaluate log density of the conditional.
        logf = log_posterior_density(model, θ) -
               logabsdetmetric(θ, sampler.geometry, model, Ap)

        for i=1:sampler.nsubsamples
            # TODO pinv can be simplified; A has orthogonal columns
            x0 = pinv(Ap) * θ

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

            logfL, θ_L = log_cond_density(L, r, c, δ, model, Ap, sampler.geometry, x0)
            while (J >0 && logfL > z)
                L = L -sampler.w
                J = J-1
                logfL, θ_L = log_cond_density(L, r, c, δ, model, Ap, sampler.geometry, x0)
            end

            logfR, θ_R = log_cond_density(R, r, c, δ, model, Ap, sampler.geometry, x0)
            while (K >0 && logfR > z)
                R = R +sampler.w
                K = K-1
                logfR, θ_R = log_cond_density(R, r, c, δ, model, Ap, sampler.geometry, x0)
            end

            # Shrinkage and sampling procedure.
            Lb = L
            Rb = R

            while (true)
                U = rand(rng, Uniform())
                α1 = Lb + U *(Rb - Lb)
                logfα1, θ_α1 = log_cond_density(α1, r, c, δ, model, Ap, sampler.geometry, x0)
                if(z < logfα1)
                    # Accept!
                    α = α1
                    logf = logfα1
                    θ[:] .= θ_α1[:]
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
        g = legendre_dual(θ, sampler.geometry, model)

        δ[j] = (r' * g) / (r' * r)

        # Save the vector
        A[:, j] .= r

    end

    return θ, θ
end
