export GaussianInverse, GaussianParam, SparseArrays
using Distributions
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
