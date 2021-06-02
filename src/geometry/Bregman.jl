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
struct DualParameter{T<:Real, G<:Bregman}<:Parameter{T}
   data::Vector{T} 
end

Array(η::DualParameter) = η.data

function DualParameter(data::Vector{T}, geometry::Bregman) where T<: Real
    return DualParameter{T, typeof(geometry)}(data)
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

    return DualParameter(gradient, geometry)
end

# TODO: legendre_dual for dual -> primal

# TODO: metric (evaluate Hessian)
