export LogDensityModel

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
    grad_log_posterior_density(model, θ)

Vector of partial derivatives of `log_posterior_density` with respect to components of `θ`.

Default uses automatic differentiation.
Alias function names: `∇logπ`.
"""
function grad_log_posterior_density(model::BayesModel, θ)
    # Default uses autodiff
    ForwardDiff.gradient(x -> log_posterior_density(model, x), θ)
end
const ∇logπ = grad_log_posterior_density

# """
#     hessian_log_posterior_density(model, θ)

# Matrix of second-order partial derivatives of `log_posterior_density` with
# respect to components of `θ`.

# Default uses automatic differentiation.
# Alias function names: `∇²logπ`.
# """
# function hessian_log_posterior_density(model::BayesModel, θ::Parameter)
#     # Default uses autodiff
#     ParameterType = Base.typename(typeof(θ)).wrapper
#     proxy(x) = log_posterior_density(model, ParameterType(x))
#     ForwardDiff.hessian(proxy, Array(θ))
# end
# const ∇²logπ = hessian_log_posterior_density


# include("Diaconis.jl")
# include("LinearGaussian.jl")
# include("Forward.jl")
# include("Rosenbrock.jl")
include("CanonicalGLM.jl")
