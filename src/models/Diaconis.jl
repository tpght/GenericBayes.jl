export ConjugateModel, DiaconisConjugate, bayesian_update

abstract type DiaconisConjugate{D<:Distribution}<:BayesModel end

struct ConjugateModel{D<:Distribution,T<:Real} <: DiaconisConjugate{D}
	n::T						# Sample size parameter
	t::Vector{T}				# Predicted mean parameter
end

size_hyperparam(model::ConjugateModel) = model.n
mean_hyperparam(model::ConjugateModel) = model.t
dimension(model::ConjugateModel) = length(model.t)

function ConjugateModel(D::Type{<:Distribution}, n::T,
						t::Vector{T}) where T<:Real
	ConjugateModel{D,T}(n,t)
end

"""
    bayesian_update

Returns a new ConjugateModel with updated hyperparameters given data.
"""
function bayesian_update(model::ConjugateModel{D}, y::Vector) where D<:Distribution
    n = length(y)
    n0 = model.n
    n1 = n + n0
    t1 = sufficient_statistics(model, y)
    ConjugateModel(D, n1, (n0 / n1) .* model.t .+ (n / n1) .* t1)
end

function log_posterior_density(model::DiaconisConjugate, θ::Vector)
	# Get model hyperparameters
	n = size_hyperparam(model)
	t = mean_hyperparam(model)

	# Evaluate log-density
	n * (θ' * t - ψ(model, θ))
end

function grad_log_posterior_density(model::DiaconisConjugate, θ::Vector)
	# Get model hyperparameters
	n = size_hyperparam(model)
	t = mean_hyperparam(model)

	# Evaluate gradient of model log-density
	n * (t - ∇ψ(model, θ))
end

function hessian_log_posterior_density(model::DiaconisConjugate, θ::Vector)
	# Get model hyperparameters
	n = size_hyperparam(model)

	# Evaluate gradient of model log-density
	-n * ∇²ψ(model, θ)
end

# TODO: Change to more general DiaconisConjugate?
function max_posterior(model::DiaconisConjugate, θ0::Vector)
	t = mean_hyperparam(model)
	∇ϕ(model, t)
end

function inverse_legendre_dual(ξ::Vector{T}, geometry::NegativeLogDensity,
                               model::DiaconisConjugate, k::Int; x0=nothing) where T <: Real
	n = size_hyperparam(model)
	t = mean_hyperparam(model)

	ξ1 = copy(ξ)
	ξ1[1:k] = ((1.0 / n) .* ξ1[1:k]) .+ model.t[1:k]

	∇ϕ(model, ξ1, k)
end

# function inverse_legendre_dual(ξ::Vector{T}, geometry::NegativeLogDensity,
#                                model::DiaconisConjugate, A::Matrix{T}; x0=nothing) where T <: Real
# 	n = size_hyperparam(model)
# 	t = mean_hyperparam(model)

# 	ξ1 = A' * (((1.0 / n) .* ξ) .+ model.t)

# 	∇ϕ(model, ξ1, A)
# end

function inverse_legendre_dual(η::Vector{T}, geometry::NegativeLogDensity,
                               model::DiaconisConjugate) where T <: Real
	n = size_hyperparam(model)
	t = mean_hyperparam(model)

	dual = ((1.0 / n) .* η) .+ model.t

	∇ϕ(model, dual)
end

# include("Multinomial.jl")
include("GaussianObservation.jl")
