import AbstractMCMC.AbstractSampler, AbstractMCMC.step
export OrthogonalGradient

"""
    OrthogonalGradient

Orthogonal Gradient sampler
"""
struct OrthogonalGradient{T<:Real} <: AbstractSampler
    geometry::Bregman           # Geometry to be used in the sampler
    proposal::Proposal
    nsubsamples::Int                  # Number of times to run the embedded sampler
    initial_θ::Vector{T}
    stop::Int
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

    # Compute gradient of Bregman generator
    g = legendre_dual(θ, sampler.geometry, model)

    for j = 1:p
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

        # scale = 1.0 / prec

        α = α0

        for i=1:sampler.nsubsamples

            scale = 1.0 / sqrt(r' * metric(θ, sampler.geometry, model) * r)
            proposal = RandomWalkProposal(scale)
            # Generate a proposal for α using α0 as initial point
            # α_prop = draw_proposal(rng, sampler.proposal, [α])[1]
            # α_prop = rand(rng, Normal(α, scale))
            α_prop = draw_proposal(rng, proposal, [α])[1]

            # Embed the proposed alpha value
            # TODO: pinv here can be speeded up (Ap has orthogonal columns)
            b = α_prop .* r  + c
            θ_prop = inverse_legendre_dual(δ[1:(j-1)], b, Ap, sampler.geometry,
            model; x0 = pinv(Ap) * θ)

            # Evaluate density at current and proposed points
            # α0 should be embedded to current θ
            log_dens_prop = log_posterior_density(model, θ_prop) -
                logabsdetmetric(θ_prop, sampler.geometry, model, Ap)
            log_dens_current = log_posterior_density(model, θ) -
                logabsdetmetric(θ, sampler.geometry, model, Ap)

            # Evaluate log-ratio of conditional proposal densities
            # logprop_ratio = log_proposal_ratio(sampler.proposal, θ_prop, θ)
            logprop_ratio = log_proposal_ratio(proposal, θ_prop, θ)
            # logprop_ratio = 0.0

            # Evaluate metropolis-hastings log-ratio
            logp_accept = log_dens_prop - log_dens_current + logprop_ratio
            # Accept/reject
            if(logp_accept >= 0 || log(rand(rng)) < logp_accept)
                # Accept
                θ = θ_prop
                α = α_prop
                continue
            end

            # If we've reached here, rejected. Keep θ the same.
        end

        # Compute gradient of Bregman generator
        g = legendre_dual(θ, sampler.geometry, model)

        δ[j] = (r' * g) / (r' * r)

        # Save the vector
        A[:, j] .= r

    end

    return θ, θ
end
