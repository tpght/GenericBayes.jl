using LinearAlgebra, ForwardDiff, Optim
export Bregman, DualParameter, NegativeLogDensity
export legendre_dual, bregman_generator

"""
    Bregman

Geometry derived from a Bregman divergence.
"""
abstract type Bregman <: Geometry end

"""
    bregman_generator

The convex function from which the Bregman divergence and geometry are derived
"""
function bregman_generator(θ::Parameter, geometry::Bregman, model::BayesModel) end

"""
    NegativeLogDensity{P}

Bregman geometry which uses the entire negative log posterior density as the generator.

Of course, the posterior density depends on parameterization. Therefore this
geometry has P<:Parameter to indicate which co-ordinates defines the geometry.
"""
struct NegativeLogDensity{P<:Parameter}<:Bregman end

function bregman_generator(θ::P, geometry::NegativeLogDensity{P2},
    model::BayesModel) where P<:Parameter where P2<:Parameter
    return -log_posterior_density(model, θ)
end

"""
    DualParameter <: Parameter

Represents the Legendre dual co-ordinate
"""
mutable struct DualParameter{T<:Real, G<:Bregman, P<:Parameter{T}}<:Parameter{T}
   components::Vector{T} 
end

Array(η::DualParameter) = η.components

function DualParameter(data::Vector{T}, geometry::Bregman, P::Type{<:Parameter{T}}) where T<: Real
    return DualParameter{T, typeof(geometry), P}(data)
end

function DualParameter(data::Vector{T}, geometry::Bregman, P::Type{<:Parameter}) where T<: Real
    return DualParameter{T, typeof(geometry), P{T}}(data)
end

"""
    legendre_dual(θ::Parameter, geometry::Bregman, model::BayesModel)

Compute the Legendre dual, i.e. the gradient of `bregman_generator`.

Default uses automatic differentiation.
"""
function legendre_dual(θ::Parameter, geometry::Bregman, model::BayesModel)
    # First compute the gradient as a vector
    ParameterType = Base.typename(typeof(θ)).wrapper
    proxy(x) = bregman_generator(ParameterType(x), geometry, model)
    gradient = ForwardDiff.gradient(proxy, θ.components)

    return DualParameter(gradient, geometry, typeof(θ))
end

"""
    legendre_dual(η::DualParameter{T, G, P}, geometry::G, model::BayesModel)

Compute the Legendre dual to η, i.e. the primal vector θ such that
legendre_dual(θ, geometry, model) = η.

Default uses automatic differentiation.
"""
function legendre_dual(η::DualParameter{T, G, P}, geometry::G,
                       model::BayesModel) where G<:Bregman where
P<:Parameter{T} where T<:Real

    # Define the function to be optimized (Legendre dual)
    ParameterType = Base.typename(P).wrapper

    # TODO Find an appropriate initial point x0
    x0 = ones(T, size(η.components, 1))

    # Define cost function
    proxy(x) = bregman_generator(ParameterType(x), geometry, model) - x' * η.components

    # Optimize the function
    # lower = lower_box(model, P)
    # upper = upper_box(model, P)
    result = optimize(proxy, x0, LBFGS(); autodiff=:forward)
    
    if(Optim.converged(result) == false)
        @show Optim.converged(result)
        @show Optim.iterations(result)
        @show Optim.iteration_limit_reached(result)
        error("Could not convert from dual to primal co-ordinates")
    end

    primal = Optim.minimizer(result)

    # Construct and return primal
    θ = P(primal)

    return θ
end

"""
    dual_bregman_generator(θ, geometry, model)

Evaluate the Legendre dual (convex conjugate) to `bregman_generator` at θ
"""
function dual_bregman_generator(θ::Parameter{T}, geometry::Bregman,
                                model::BayesModel) where T<:Real
    η = legendre_dual(θ, geometry, model)
    return θ.components' * η.components - bregman_generator(θ, geometry, model)
end

"""
    metric(θ::Parameter{T}, geometry::Bregman, model::BayesModel)

Compute the Riemannian metric, i.e. the hessian of `bregman_generator`.
"""
function metric(θ::Parameter{T}, geometry::Bregman, model::BayesModel) where T<:Real
    # Default uses autodiff
    ParameterType = Base.typename(typeof(θ)).wrapper
    proxy(x) = bregman_generator(ParameterType(x), geometry, model)
    ForwardDiff.hessian(proxy, θ.components)
end

"""
    metric(η::DualParameter{T, G, P}, geometry::Bregman, model::BayesModel)

Compute the Riemannian metric, i.e. the inverse hessian of `bregman_generator`.
"""
function metric(η::DualParameter{T, G, P}, geometry::G,
                       model::BayesModel) where G<:Bregman where
P<:Parameter{T} where T<:Real
    # Default uses autodiff
    ParameterType = Base.typename(P).wrapper

    # For now, literally just do the inversion
    θ = legendre_dual(η, geometry, model)
    my_metric = metric(θ, geometry, model)

    inv(my_metric)
end
