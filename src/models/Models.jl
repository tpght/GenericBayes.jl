export LogDensityModel, ReversedModel

"""
    prior

Returns the prior in a particular co-ordinate system as a Distributions.Distribution
"""
function prior(model::BayesModel) end

"""
    log_posterior_density(model, θ)

The (natural) log of the posterior probability density of `model` up to additive constant.

Alias function names: `lpd` and `logπ`.
"""
function log_posterior_density(model::BayesModel, θ)
    logpdf(prior(model), θ) + loglikelihood(model, θ)
end
const lpd = log_posterior_density
const logπ = log_posterior_density

"""
    LogDensityModel

Simplest type of model; defines a log-density function.
"""
struct LogDensityModel <: BayesModel
    log_density::Function
    dimension::Int
end

log_posterior_density(model::LogDensityModel, θ) = model.log_density(θ)
dimension(model::LogDensityModel) = model.dimension

"""
    ReversedModel

Reverses the labelling of the parameters of another model.
"""
struct ReversedModel <: BayesModel
    model::BayesModel
end

function log_posterior_density(model::ReversedModel, θ)
    log_posterior_density(model.model, reverse(θ))
end
dimension(model::ReversedModel) = dimension(model.model)

"""
    grad_log_posterior_density(model, θ)

Vector of partial derivatives of `log_posterior_density` with respect to components of `θ`.

Default uses automatic differentiation.
Alias function names: `∇logπ`.
"""
function grad_log_posterior_density(model::BayesModel, θ)
    # Default uses autodiff
    ForwardDiff.gradient(x -> log_posterior_density(model, x), θ)
end

function grad_log_posterior_density(model::BayesModel, θ, k::Int)
    g = grad_log_posterior_density(model, θ)
    g[1:k]
end

function grad_log_posterior_density(model::BayesModel, θ::Vector{T},
    A::Matrix{T}) where D<:Bernoulli where T<:Real

    A' * grad_log_posterior_density(model, θ)
end

const ∇logπ = grad_log_posterior_density

"""
    hessian_log_posterior_density(model, θ)

Matrix of second-order partial derivatives of `log_posterior_density` with
respect to components of `θ`.

Default uses automatic differentiation.
Alias function names: `∇²logπ`.
"""
function hessian_log_posterior_density(model::BayesModel, θ)
    # Default uses autodiff
    proxy(x) = log_posterior_density(model, x)
    ForwardDiff.hessian(proxy, θ)
end
const ∇²logπ = hessian_log_posterior_density

function hessian_log_posterior_density(model::BayesModel, θ, k::Int)
    h = hessian_log_posterior_density(model, θ)
    h[1:k, 1:k]
end

function hessian_log_posterior_density(model::BayesModel, θ::Vector{T},
    A::Matrix{T}) where T<:Real
    h = hessian_log_posterior_density(model, θ)
    A' * h * A
end

include("Diaconis.jl")
# include("LinearGaussian.jl")
# include("Forward.jl")
# include("Rosenbrock.jl")
include("CanonicalGLM.jl")
include("ChenIbrahimGLM.jl")
include("SimpleGaussian.jl")
include("GaussianInverse.jl")
include("TransformedModel.jl")
