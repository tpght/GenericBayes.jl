import AbstractMCMC.AbstractSampler, Base.show, AbstractMCMC.step

export stretch_move!, swap_state!, leapfrog!
export ProductHMCState, ProductManifoldHMC
export hamiltonian, grad_hamiltonian, step

"""
    ProductManifoldHMC

Hamiltonian Monte Carlo on the product space Q x Q, sampling product density

Described in chapter 6 of the thesis.
"""
struct ProductManifoldHMC{T<:Real, P<:Parameter{T}} <: AbstractSampler
    geometry::Bregman           # Bregman geometry to use
    ϵ::T                        # Leapfrog step-size
    L::Int                      # Number of Leapfrog steps to take
    stretch_parameter::T        # The `a` parameter in the stretch move
end

"""
    step(rng, model::BayesModel, sampler::ProductManifoldHMC,
              state=nothing; kwargs...)

One iteration of the product manifold HMC method.
"""
function step(rng, model::BayesModel, sampler::ProductManifoldHMC{T, P},
              current_state=nothing; kwargs...) where P<:Parameter{T} where T<:Real

    # First, generate an initial state if required
    if (current_state == nothing)
        θ0 = max_posterior(model, P(ones(dimension(model))))
        θ1 = deepcopy(θ0)

        # Move both points OFF the mode
        # The mode of the distribution is a fixed point of the dynamics...
        θ1.components = θ1.components + rand(rng, dimension(model))
        θ0.components = θ0.components + rand(rng, dimension(model))
        
        η0 = legendre_dual(θ1, sampler.geometry, model)
        state = ProductHMCState(θ0, η0)

        return state.primal.components, state
    end

    # Sample using ProductHMC method
    proposal_state = deepcopy(current_state)

    # Save state
    H_current = hamiltonian(current_state, sampler.geometry, model)
    
    # Walk move in primal co-ordinates
    θ2 = legendre_dual(proposal_state.dual, sampler.geometry, model)
    logπ_old = logπ(model, θ2)
    z = walk_move!(θ2, proposal_state.primal, rng, sampler.stretch_parameter)
    proposal_state.dual = legendre_dual(θ2, sampler.geometry, model)

    # Integrate using a random number of leapfrog steps
    for i in 1:rand(1:sampler.L)
        leapfrog!(proposal_state, sampler.ϵ, sampler.geometry, model)
    end

    # Swap primal and dual so that proposal is symmetric
    swap_state!(proposal_state, sampler.geometry, model)

    # Metropolis-Hastings accept / reject
    H_proposal = hamiltonian(proposal_state, sampler.geometry, model)
    logp = H_current - H_proposal        # Log acceptance probability

    if((H_current > H_proposal) || rand(rng) < exp(logp))
        # Accept
        return proposal_state.primal.components, proposal_state
    end

    # For now, return primal components
    return current_state.primal.components, current_state
end

"""
    walk_move(θ1, θ2,rng,a)

Implementation of the affine-invariant stretch move.
"""
function walk_move!(θ1::P, θ2::P, rng::AbstractRNG=Random.GLOBAL_RNG, a=2.0) where P<:Parameter{T} where T<:Real
    # Inverse CDF for the z varianble in the stretch move.
    # This allows sampling using the stretch move.
    invcdf(u) = (u.*(a-1) .+ 1).^2 ./ a

    # Sample the z variable using the inverse CDF defined above
    z = invcdf(rand(rng, length(θ1.components)))

    # Update θ1 using the stretch move
    θ1.components = θ2.components .+ z .* (θ1.components .- θ2.components)

    return z
end

"""
    ProductHMCState{P} where P<:Parameter{T} where T<:Real

Represents a primal-dual pair, i.e. a state in the product-manifold
"""
mutable struct ProductHMCState{T<:Real, G<:Bregman, P<:Parameter{T}}
    primal::P
    dual::DualParameter{T,G,P}
end

ProductHMCState(θ1::P, θ2::P, geometry::Bregman, model::BayesModel) where
    P<:Parameter{T} where T<:Real =
    ProductHMCState(θ1, legendre_dual(θ2, geometry, model))

function show(io::IO, state::ProductHMCState)
    @show state.primal.components
    @show state.dual.components
end

"""
    leapfrog!(state::ProductHMCState, sampler::ProductManifoldHMC, model::BayesModel)

The leapfrog integrator used to integrate the Hamiltonian dynamics.
"""
function leapfrog!(state::ProductHMCState, ϵ::Real, geometry::Bregman,
                   model::BayesModel)

    # Half-step in dual
    dHdθ, dHdη = grad_hamiltonian(state, geometry, model)
    state.dual.components = state.dual.components - 0.5 * ϵ * dHdθ

    # Full-step in primal
    dHdθ, dHdη = grad_hamiltonian(state, geometry, model)
    state.primal.components = state.primal.components + ϵ * dHdη

    # Half-step in dual
    dHdθ, dHdη = grad_hamiltonian(state, geometry, model)
    state.dual.components = state.dual.components - 0.5 * ϵ * dHdθ
end

"""
    swap_state!(state, geometry, model)

Swaps the primal-dual representation of points
"""
function swap_state!(state::ProductHMCState{T,G,P}, geometry::G,
                     model::BayesModel) where P<:Parameter{T} where G<:Bregman where T<:Real

    # Make a copy of the primally represented point
    original_primal = P(state.primal.components)

    # Convert the dually-represented point to primal
    state.primal = legendre_dual(state.dual, geometry, model)

    # Convert the primally-represented point to dual
    state.dual = legendre_dual(original_primal, geometry, model)

end

"""
    hamiltonian(state::ProductHMCState{T,G,P}, geometry::G, model::BayesModel)

Energy function on the product manifold, i.e. independent primal/dual target
"""
function hamiltonian(state::ProductHMCState{T,G,P}, geometry::G,
                     model::BayesModel) where G<:Bregman where P<:Parameter{T} where T<:Real

    # H(θ, η) = -logπ(θ) -logπ(∇φ(η)) + logdet(G(∇φ(η)))

    H = -logπ(model, state.primal)

    # Convert dual co-ordinate to primal
    θ_dual = legendre_dual(state.dual, geometry, model)

    # Compute density of dual co-ordinate, including jacobian term
    H = H - logπ(model, θ_dual)
    H = H + logabsdetmetric(θ_dual, geometry, model)

    return H
end

"""
    grad_hamiltonian(state::ProductHMCState{T,G,P}, geometry::G, model::BayesModel)

Evaluate gradient of `hamiltonian` with respect to primal and dual variables.
"""
function grad_hamiltonian(state::ProductHMCState{T,G,P}, geometry::G,
                          model::BayesModel) where G<:Bregman where
P<:Parameter{T} where T<:Real
    
    dHdθ = -∇logπ(model, state.primal)

    θ2 = legendre_dual(state.dual, geometry, model)
    dHdθ2 = -∇logπ(model, θ2)

    dHdθ2 = dHdθ2 + grad_logabsdetmetric(θ2, geometry, model)

    Gθ2 = metric(θ2, geometry, model)
    dHdη = inv(Gθ2) * dHdθ2

    return dHdθ, dHdη
end
