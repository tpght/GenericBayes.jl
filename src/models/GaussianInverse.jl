export GaussianInverse, GaussianParam, circular_model, precondition_model
using Distributions, SparseArrays
import Base.map

struct GaussianInverse{T<:Real, P} <: BayesModel
    Λ::Symmetric{T, <:AbstractArray{T,P}}   # Covariance
    w::Vector{T}                # Right-hand side
end

dimension(model::GaussianInverse) = length(model.w)
covariance(model::GaussianInverse) = model.Λ

"""
circular_model(p::Int, ϕ::Real, κ::Real)

Arguments:
- `ϕ`: ∈ [0, 1) coupling of neighbours
- `κ`: Precision at each site.
 Creates the model from Example 13.2.1 in Handbook of Spatial Statistics
"""
function circular_model(p::Int, ϕ::T, κ::T, w::Vector{T}) where T<:Real
    circular_model(p, ϕ, κ, w, spdiagm(ones(p)))
end


"""
circular_model(p::Int, ϕ::Real, κ::Real, U::AbstractArray)

Arguments:
- `ϕ`: ∈ [0, 1) coupling of neighbours
- `κ`: Precision at each site.
 Creates the model from Example 13.2.1 in Handbook of Spatial Statistics
"""
function circular_model(p::Int, ϕ::T, κ::T, w::Vector{T}, U::AbstractArray) where T<:Real
    Q = spdiagm(Pair(-1, -ϕ * ones(p-1)),
                Pair(1, -ϕ * ones(p-1)),
                Pair(0, 2.0 * ones(p)))
    Q[p,1] = -ϕ
    Q[1,p] = -ϕ
    Q = (0.5 * κ) .* Q

    GaussianInverse(Symmetric(U' * Q * U), U' * w)
end
