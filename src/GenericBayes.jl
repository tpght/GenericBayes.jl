module GenericBayes

using StatsBase, LinearAlgebra, ForwardDiff
import Base.Vector, Base.map, Base.Array, Base.length

export Multinomial
export log_posterior_density, sufficient_statistic
export ∇logπ, grad_log_posterior_density
export ∇²logπ, hessian_log_posterior_density
export likelihood, loglikelihood, mle, map, prior_mode, prior, simulate

export @vector_param, @reparam

"""
    BayesModel

A `BayesModel` is any type representing a Bayesian model.

Any `BayesModel` should implement at least `log_posterior_density`.
"""
abstract type BayesModel end

"""
    Parameter{T<:Real}

A `Parameter` is any type representing a realization of the posterior distribution.

For Bayesian models, this corresponds to parameters of a statistical family.
"""
abstract type Parameter{T<:Real} end

"""
    vector_param(name)

Generates code that defines a `Parameter` called `$name`.

The generated code implements `Base.Array`, `Base.Vector`, and `Base.length` which returns the dimension of the vector.
"""
macro vector_param(name)
    def =:(
        struct $name{T} <: Parameter{T}
            components::Vector{T}
        end
    )
    ar = :( Array(θ::$name) = θ.components )
    vec = :( Vector(θ::$name) = Array(θ) )
    len = :( length(θ::$name) = length(θ.components) )
    return quote
        $(esc(def))
        $(esc(ar))
        $(esc(vec))
        $(esc(len))
    end
end

"""
    reparam(from, to, f)
"""
macro reparam(from, to, f)
    return :(
    function $to(θ::$from, model::Union{BayesModel,Nothing}=nothing)
        return $to($f(model, θ.components))
    end
    )
end


"""
    lower_box(model, P)

Define the lower bounds for components of parameters of type `P` when used
to parameterize `model`.

Default returns `-Inf` in each component, i.e. no lower bounds.
"""
lower_box(model::BayesModel, P::Type{<:Parameter}) = [-Inf for i in 1:dimension(model)]

"""
    upper_box(model, P)

Define the upper bounds for components of parameters of type `P` when used
to parameterize `model`.

Default returns `Inf` in each component, i.e. no upper bounds.
"""
upper_box(model::BayesModel, P::Type{<:Parameter}) = [Inf for i in 1:dimension(model)]

# TODO
# jacobian(model::BayesModel, θ::Parameter)
""" Log absolute-value of the determinant of the jacobian matrix of a change of parameterization. """
logabsdetjac(model::BayesModel, θ::Parameter) = log(abs(det(jacobian())))

"""
    log_posterior_density(model, θ)

The (natural) log of the posterior probability density of `model` up to additive constant.

Alias function names: `lpd` and `logπ`.
"""
function log_posterior_density(model::BayesModel, θ::Parameter) end
const lpd = log_posterior_density
const logπ = log_posterior_density

"""
    grad_log_posterior_density(model, θ)

Vector of partial derivatives of `log_posterior_density` with respect to components of `θ`.

Default uses automatic differentiation.
Alias function names: `∇logπ`.
"""
function grad_log_posterior_density(model::BayesModel, θ::Parameter)
    # Default uses autodiff
    ParameterType = Base.typename(typeof(θ)).wrapper
    proxy(x) = log_posterior_density(model, ParameterType(x))
    ForwardDiff.gradient(proxy, Array(θ))
end
const ∇logπ = grad_log_posterior_density

"""
    hessian_log_posterior_density(model, θ)

Matrix of second-order partial derivatives of `log_posterior_density` with respect to components of `θ`.

Default uses automatic differentiation.
Alias function names: `∇²logπ`.
"""
function hessian_log_posterior_density(model::BayesModel, θ::Parameter)
    # Default uses autodiff
    ParameterType = Base.typename(typeof(θ)).wrapper
    proxy(x) = log_posterior_density(model, ParameterType(x))
    ForwardDiff.hessian(proxy, Array(θ))
end
const ∇²logπ = hessian_log_posterior_density

"""
    mle(model, θ, data)

the maximum likelihood estimate (mle) of `model` when parameterized by `typeof(θ)`
and passing `data` to `loglikelihood`.

Optimization uses `θ` as an initial guess.
"""
function mle(model::BayesModel, θ::Parameter, data::Array)
    ParameterType = Base.typename(typeof(θ)).wrapper
    cost(x) = -loglikelihood(model, ParameterType(x), data)
    od = OnceDifferentiable(cost, Array(θ); autodiff=:forward)
    lower = lower_box(model, ParameterType)
    upper = upper_box(model, ParameterType)
    x_min = Optim.minimizer(optimize(od, lower, upper, Array(θ), Fminbox(BFGS())))
    ParameterType(x_min)
end

"""
    mle(model, θ)

The maximum likelihood estimate (MLE) of `model` when parameterized by `typeof(θ)`
and using data already specified in `model`.

Optimization uses `θ` as an initial guess.
"""
function mle(model::BayesModel, θ::Parameter)
    ParameterType = Base.typename(typeof(θ)).wrapper
    cost(x) = -loglikelihood(model, ParameterType(x))
    od = OnceDifferentiable(cost, Array(θ); autodiff=:forward)
    lower = lower_box(model, ParameterType)
    upper = upper_box(model, ParameterType)
    x_min = Optim.minimizer(optimize(od, lower, upper, Array(θ), Fminbox(BFGS())))
    ParameterType(x_min)
end

"""
    map(model, θ, data)

The maximum-a-posterior (map) of `model` when parameterized by `typeof(θ)`
and passing `data` to `loglikelihood`.

Optimization uses `θ` as an initial guess.
"""
function map(model::BayesModel, θ::Parameter, data::Array)
    ParameterType = Base.typename(typeof(θ)).wrapper
    cost(x) = -log_posterior_density(model, ParameterType(x), data)
    td = OnceDifferentiable(cost, Array(θ); autodiff=:forward)
    lower = lower_box(model, ParameterType)
    upper = upper_box(model, ParameterType)
    x_min = Optim.minimizer(optimize(od, lower, upper, Array(θ), Fminbox(BFGS())))
    ParameterType(x_min)
end

"""
    map(model, θ, data)

The maximum-a-posterior (map) of `model` when parameterized by `typeof(θ)`
and using data internally specified in `model`.

Optimization uses `θ` as an initial guess.
"""
function map(model::BayesModel, θ::Parameter)
    ParameterType = Base.typename(typeof(θ)).wrapper
    cost(x) = -log_posterior_density(model, ParameterType(x))
    od = OnceDifferentiable(cost, Array(θ); autodiff=:forward)
    lower = lower_box(model, ParameterType)
    upper = upper_box(model, ParameterType)
    x_min = Optim.minimizer(optimize(od, lower, upper, Array(θ), Fminbox(BFGS())))
    ParameterType(x_min)
end

"""
    check_param(model, θ)

Returns True if `θ` is a permissible parameter for `model`, i.e.
lies in the parameter space of `model`.
"""
function check_param(model::BayesModel, θ::Parameter)
    # Check parameter has the right dimension
    # NOTE: This check might fail unnecessarily;
    # e.g. a probability parameter has length n+1
    length(θ) == dimension(model) ? nothing : throw(ArgumentError("Parameter has dimension $(length(θ)); model is dimension $(dimension(θ))"))
    
    # Check parameter is in bounds
    ParameterType = Base.typename(typeof(θ)).wrapper
    p = Vector(θ)
    all((p .> lower_box(model, ParameterType)) .& (p .< upper_box(model, ParameterType))) ? nothing : throw(ArgumentError("Parameter is out of bounds: $(Vector(θ))"))
end

include("models/ExponentialFamilies.jl")
include("models/LinearGaussian.jl")
include("models/Forward.jl")
include("vis/Density.jl")

end # module
