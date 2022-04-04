export GaussianInverse, GaussianParam, circular_model
using Distributions, SparseArrays
import Base.map

struct GaussianInverse{T<:Real, P} <: BayesModel
    Λ::AbstractArray{T,P}   # Covariance
    w::Vector{T}                # Right-hand side
end

dimension(model::GaussianInverse) = length(model.w)
covariance(model::GaussianInverse) = model.Λ

# function max_posterior(model::GaussianInverse, θ0)
#     model.Λ \ model.w
# end

# function sample(model::GaussianInverse, θ0, nsamples)
#     rand(MvNormal(map(model, θ), covariance(model)), nsamples)
# end

# Creates the model from Example 13.2.1 in Handbook of Spatial Statistics
"""
circular_model(p::Int, ϕ::Real, κ::Real)

Arguments:
- `ϕ`: ∈ [0, 1) coupling of neighbours
- `κ`: Precision at each site.
"""
function circular_model(p::Int, ϕ::T, κ::T, w::Vector{T}) where T<:Real
    Q = spdiagm(Pair(-1, -ϕ * ones(p-1)),
                Pair(1, -ϕ * ones(p-1)),
                Pair(0, 2.0 * ones(p)))
    Q[p,1] = -ϕ
    Q[1,p] = -ϕ
    Q = (0.5 * κ) .* Q

    GaussianInverse(Q, w)
end
