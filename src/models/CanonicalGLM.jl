export inverse_canonical_link, CanonicalGLM, GLM

"""
    GLM

Generalized linear model. Parameter is the exponential family from which
observations are taken. A Gaussian prior is used on the coefficients.
"""
abstract type GLM{D<:Distribution}<:BayesModel end
struct CanonicalGLM{D, T<:Real}<:GLM{D}
    X::Array{T, 2}              # n x p Design matrix
    y::Array{T, 1}              # n x 1 data
    μ_p::Array{T, 1}        # p x 1 prior mean
    Σ_p::Array{T, 2}        # p x p prior covariance
end

dimension(model::CanonicalGLM) = size(model.X)[2]

"""
    inverse_canonical_link

Returns the mean of the data for a given linear combination θ of covariates.
"""
function inverse_canonical_link(D::Type{<:Distribution}, θ::Real) end

inverse_canonical_link(::Type{Poisson}, θ::Real) = exp(θ)
inverse_canonical_link(::Type{Bernoulli}, θ::Real) = logistic(θ)

function prior(model::CanonicalGLM)
    MvNormal(model.μ_p, model.Σ_p)
end

function loglikelihood(model::CanonicalGLM{D}, β) where D<:Distribution
    linear_comb = model.X * β
    mean = inverse_canonical_link.(D, linear_comb)
    data_generator = loglikelihood(Product(D.(mean)), model.y)
end