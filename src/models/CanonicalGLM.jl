export inverse_canonical_link, CanonicalGLM, GLM

"""
    GLM

Generalized linear model. Parameter is the exponential family from which
observations are taken. A Gaussian prior is used on the coefficients.
"""
abstract type GLM{D<:Distribution}<:BayesModel end

struct CanonicalGLM{D<:Distribution, T<:Real}<:GLM{D}
    X::Array{T, 2}              # n x p Design matrix
    y::Array{T, 1}              # n x 1 data
    μ_p::Array{T, 1}        # p x 1 prior mean
    Σ_p::Array{T, 2}        # p x p prior covariance
    Q_p::Array{T, 2}        # p x p prior precision (cached)
    function CanonicalGLM(X::Matrix{T}, y::Vector, μ_p::Vector{T},
    Σ_p::Matrix{T}, dist::Type{D}) where T<:Real where D<:Distribution
        # Compute and cache the precision = inverse of covariance
        new{D,T}(X, y, μ_p, Σ_p, inv(Σ_p))
    end
end
dimension(model::CanonicalGLM) = size(model.X)[2]

"""
    inverse_canonical_link

Returns the mean of the data for a given linear combination θ of covariates.
"""
function inverse_canonical_link(D::Type{<:Distribution}, θ::Real) end

inverse_canonical_link(::Type{Poisson}, θ::Real) = exp(θ)
inverse_canonical_link(::Type{Bernoulli}, θ::Real) = logistic(θ)

function prior(model::CanonicalGLM{D,T}) where D<:Distribution where T<:Real
    # NOTE: does Distributions cache precision inside the MultivariateNormal
    # struct type, so that I don't have to do it myself?
    MultivariateNormal(model.μ_p, model.Σ_p)
end

function loglikelihood(model::CanonicalGLM{D,T}, β) where D<:Distribution where
    T<:Real
    linear_comb = model.X * β
    mean = inverse_canonical_link.(D, linear_comb)
    data_generator = loglikelihood(Product(D.(mean)), model.y)
end

function log_posterior_density(model::CanonicalGLM{D,T}, β) where D<:Bernoulli where T<:Real
    linear_comb = model.X * β
    # TODO Need to compute a fast log(1 + exp(...))
    ll=model.y' * model.X * β - sum(log.( 1.0 .+ exp.(linear_comb)))
    lp=-0.5 * (β .- model.μ_p)' * model.Q_p * (β .- model.μ_p)
    ll + lp
end

function grad_log_posterior_density(model::CanonicalGLM{D,T}, β) where D<:Bernoulli where T<:Real
    linear_comb = model.X * β
    pr = logistic.(linear_comb)
    model.X' * (model.y - pr) - model.Q_p * (β .- model.μ_p)
end

function grad_log_posterior_density(model::CanonicalGLM{D,T}, β, k::Int) where D<:Bernoulli where T<:Real
    linear_comb = model.X * β
    pr = logistic.(linear_comb)
    Xp = model.X[:, 1:k]
    Xp' * (model.y - pr) - model.Q_p[1:k, :] * (β .- model.μ_p)
end

function grad_log_posterior_density(model::CanonicalGLM{D,T}, β::Vector{T},
    A::Matrix{T}) where D<:Bernoulli where T<:Real

    linear_comb = model.X * β
    pr = logistic.(linear_comb)
    Xp = model.X * A
    Xp' * (model.y - pr) - A' * model.Q_p * (β .- model.μ_p)
end

function hessian_log_posterior_density(model::CanonicalGLM{D,T}, β) where D<:Bernoulli where T<:Real
    linear_comb = model.X * β
    pr = logistic.(linear_comb)
    λ = -pr .* (1.0 .- pr)
    model.X' * (model.X .* λ) - model.Q_p
end

function hessian_log_posterior_density(model::CanonicalGLM{D,T}, β, k::Int) where
    D<:Bernoulli where T<:Real
    linear_comb = model.X * β
    pr = logistic.(linear_comb)
    λ = -pr .* (1.0 .- pr)
    Xp = model.X[:, 1:k]
    Xp' * (Xp .* λ) - model.Q_p[1:k, 1:k]
end

function hessian_log_posterior_density(model::CanonicalGLM{D,T}, β::Vector{T}, A::Matrix{T}) where
    D<:Bernoulli where T<:Real

    linear_comb = model.X * β
    pr = logistic.(linear_comb)
    λ = -pr .* (1.0 .- pr)
    Xp = model.X * A
    Xp' * (Xp .* λ) - A' * model.Q_p * A
end
