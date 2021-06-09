import AbstractMCMC.AbstractSampler, Base.show

export stretch_move!, swap_state!
export ProductHMCState
export hamiltonian, grad_hamiltonian

"""
    ProductManifoldHMC

Hamiltonian Monte Carlo on the product space Q x Q, sampling product density

Described in chapter 6 of the thesis.
"""
mutable struct ProductManifoldHMC{T<:Real} <: AbstractSampler
    geometry::Bregman           # Bregman geometry to use
    ϵ::T                        # Leapfrog step-size
    L::Int                      # Number of Leapfrog steps to take
    stretch_parameter::T        # The `a` parameter in the stretch move
end

"""
    stretch_move(θ1, θ2,rng,a)

Implementation of the affine-invariant stretch move.
"""
# function stretch_move!(θ1::P, θ2::P, rng::AbstractRNG, a=2.0) where P<:Parameter{T} where T<:Real
#     # Inverse CDF for the z varianble in the stretch move.
#     # This allows sampling using the stretch move.
#     invcdf(u) = (u*(a-1) + 1)^2 / a

#     # Sample the z variable using the inverse CDF defined above
#     z = invcdf(rand(rng, Uniform()))

#     # Update θ1 using the stretch move
#     θ1.components = θ2.components + z * (θ1.components - θ2.components)
# end

"""
    stretch_move(θ1, θ2)

Implementation of the affine-invariant stretch move.
"""
function stretch_move!(θ1::P, θ2::P, a = 2.0) where P<:Parameter{T} where T<:Real
    # Inverse CDF for the z varianble in the stretch move.
    # This allows sampling using the stretch move.
    invcdf(u) = (u*(a-1) + 1)^2 / a

    # Sample the z variable using the inverse CDF defined above
    z = invcdf(rand(Uniform()))

    # Update θ1 using the stretch move
    θ1.components = θ2.components + z * (θ1.components - θ2.components)
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
    leapfrog!(state::ProductHMCState, sampler::ProductManifoldHMC)

The leapfrog integrator used to integrate the Hamiltonian dynamics.
"""
function leapfrog!(state::ProductHMCState, sampler::ProductManifoldHMC,
                   model::BayesModel)
    ϵ = sampler.ϵ
    dHdθ, dHdη = grad_hamiltonian(state, sampler.geometry, model)

    # Half-step in primal
    state.primal.components = state.primal.components + 0.5 * ϵ * dHdη

    # Full-step in dual
    dHdθ, dHdη = grad_hamiltonian(state, sampler.geometry, model)
    state.dual.components = state.primal.components - ϵ * dHdθ

    # Half-step in primal
    dHdθ, dHdη = grad_hamiltonian(state, sampler.geometry, model)
    state.primal.components = state.primal.components + 0.5 * ϵ * dHdη
end

"""
    swap_state!(state::ProductHMCState{T,G,P})

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

    jac = metric(θ_dual, geometry, model)
    logabsdetjac = log(abs(det(jac)))
    H = H + logabsdetjac

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


    PrimalType = Base.typename(P).wrapper
    proxy(x) = log(abs(det(metric(PrimalType(x), geometry, model))))
    dHdθ2 = dHdθ2 + ForwardDiff.gradient(proxy, θ2.components)

    Gθ2 = metric(θ2, geometry, model)
    dHdη = inv(Gθ2) * dHdθ2

    return dHdθ, dHdη
end
