module GenericBayes

using StatsBase, LinearAlgebra, ForwardDiff
import Base.Vector, Base.map, Base.Array

export Multinomial
export log_posterior_density, sufficient_statistic
export ∇logπ, grad_log_posterior_density
export ∇²logπ, hessian_log_posterior_density
export likelihood, loglikelihood, mle, map, prior_mode, prior, simulate

export @vector_param, @reparam

"""
    BayesModel

Any type representing a Bayesian model.
"""
abstract type BayesModel end

"""
    Parameter{T<:Real}

Any type representing a model parameter.
"""
abstract type Parameter{T<:Real} end

""" Macro creates a new vector parameter type """
macro vector_param(name)
    def =:(
        struct $name{T} <: Parameter{T}
            components::Vector{T}
        end
    )
    ar = :( Array(θ::$name) = θ.components )
    vec = :( Vector(θ::$name) = Array(θ) )
    return quote
        $(esc(def))
        $(esc(ar))
        $(esc(vec))
    end
end

""" Macro for defining a relationship between co-ordinates """
macro reparam(from, to, f)
    return :(
    function $to(θ::$from, model::Union{BayesModel,Nothing}=nothing)
        return $to($f(model, θ.components))
    end
    )
end

# TODO
# jacobian(model::BayesModel, θ::Parameter)
""" Log absolute-value of the determinant of the jacobian matrix of a change of parameterization. """
logabsdetjac(model::BayesModel, θ::Parameter) = log(abs(det(jacobian())))

""" Log posterior density of the model. """
function log_posterior_density(model::BayesModel, θ::Parameter) end
const lpd = log_posterior_density
const logπ = log_posterior_density

""" Gradient of log posterior density. """
function grad_log_posterior_density(model::BayesModel, θ::Parameter)
    # Default uses autodiff
    ParameterType = Base.typename(typeof(θ)).wrapper
    proxy(x) = log_posterior_density(model, ParameterType(x))
    ForwardDiff.gradient(proxy, Array(θ))
end
const ∇logπ = grad_log_posterior_density

""" Hessian of log posterior density. """
function hessian_log_posterior_density(model::BayesModel, θ::Parameter)
    # Default uses autodiff
    ParameterType = Base.typename(typeof(θ)).wrapper
    proxy(x) = log_posterior_density(model, ParameterType(x))
    ForwardDiff.hessian(proxy, Array(θ))
end
const ∇²logπ = hessian_log_posterior_density

""" The maximum likelihood estimate (MLE) of the model. """
function mle(model::BayesModel, θ::Parameter)
    ParameterType = Base.typename(typeof(θ)).wrapper
    cost(x) = -loglikelihood(model, ParameterType(x))
    td = TwiceDifferentiable(cost, Array(θ); autodiff=:forward)
    x_min = Optim.minimizer(optimize(td, Array(θ), Newton()))
    ParameterType(x_min)
end

""" The maximum-a-posterior (MAP) of the model. """
function map(model::BayesModel, θ::Parameter)
    ParameterType = Base.typename(typeof(θ)).wrapper
    cost(x) = -log_posterior_density(model, ParameterType(x))
    td = TwiceDifferentiable(cost, Array(θ); autodiff=:forward)
    x_min = Optim.minimizer(optimize(td, Array(θ), Newton()))
    ParameterType(x_min)
end

include("models/ExponentialFamilies.jl")
include("models/LinearGaussian.jl")
include("models/Forward.jl")
include("vis/Density.jl")

end # module
