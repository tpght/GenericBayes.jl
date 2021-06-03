export metric

"""
    Geometry

Parent type for different geometries.
"""
abstract type Geometry end

"""
    metric(θ::Parameter{T}, geometry::Bregman, model::BayesModel)

Compute the Riemannian metric tensor in co-ordinate system defined by Parameter
"""
function metric(θ::Parameter{T}, geometry::Geometry, model::BayesModel) where
    T<:Real end

# Include files implementing geometries
include("Bregman.jl")
