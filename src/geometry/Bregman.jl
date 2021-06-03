using LinearAlgebra, ForwardDiff
export Bregman, DualParameter
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
function bregman_generator(θ::Parameter, geometry::Bregman, model::BayesModel)
    error("unimplemented")
end

"""
    DualParameter <: Parameter

Represents the Legendre dual co-ordinate
"""
struct DualParameter{T<:Real, G<:Bregman, P<:Parameter{T}}<:Parameter{T}
   components::Vector{T} 
end

Array(η::DualParameter) = η.components

function DualParameter(data::Vector{T}, geometry::Bregman, P::Type{<:Parameter{T}}) where T<: Real
    return DualParameter{T, typeof(geometry), P}(data)
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

# TODO: legendre_dual for dual -> primal
function legendre_dual(η::DualParameter{T, G, P}, geometry::G,
                       model::BayesModel) where G<:Bregman where
P<:Parameter{T} where T<:Real

    # Define the function to be optimized (Legendre dual)
    ParameterType = Base.typename(P).wrapper
    proxy(x) = bregman_generator(ParameterType(x), geometry, model) - x' * η.components

    # Optimize the function
    # TODO Find an appropriate initial point x0
    x0 = ones(T, size(η.components, 1))
    result = optimize(proxy, x0, LBFGS(); autodiff = :forward)
    primal = Optim.minimizer(result)

    # Construct and return primal
    θ = P(primal)

    return θ
end

# TODO: metric (evaluate Hessian)
