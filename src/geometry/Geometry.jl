export metric, logabsdetmetric

"""
    Geometry

Parent type for different geometries.
"""
abstract type Geometry end

"""
    metric(θ, geometry::Bregman, model::BayesModel)

Compute the Riemannian metric tensor
"""
function metric(θ, geometry::Geometry, model::BayesModel) where T<:Real end

include("Bregman.jl")
include("InheritedBregman.jl")
