export MultinomialConjugate, ProbabilityParameter

""" Represents a multinomial likelihood with conjugate prior. """
struct MultinomialConjugate <: DiaconisConjugate
	size::Int		# Number of outcomes
	np::Int		# Prior sample size
	data::Vector{Int}	# Vector of categorical data
	tp::Vector{Float64}	# Prior hyperparameters
end

function diaconis_hyperparam(model::MultinomialConjugate)
	model.tp
end

@vector_param ProbabilityParameter
@reparam NaturalParameter ProbabilityParameter (m,x)->[1.0 / (1.0 + sum(exp.(x))); exp.(x) / (1.0 + sum(exp.(x)))]
@reparam ProbabilityParameter NaturalParameter (m,x)->log.(x[2:end] ./ x[1])

function sufficient_statistic(m::MultinomialConjugate)
	# EXCLUDE 1 in the frequencies data
	# (remember model has two degrees of freedom)
	freqs = counts(m.data, 2:(m.size))
	freqs ./ length(m.data)
end

function sufficient_statistic(model::MultinomialConjugate, y::Vector{Int})
	freqs = counts(y, 2:model.size)
	freqs ./ length(y)
end

ψ(θ::NaturalParameter) = log(1.0 + sum(exp.(θ.comps)))
∇ψ(model::MultinomialConjugate, θ::NaturalParameter) = exp.(θ.comps) ./ (1.0 + sum(exp.(θ.comps)))
sample_size(model::MultinomialConjugate) = length(model.data)
prior_sample_size(model::MultinomialConjugate) = model.np

function log_posterior_density(model::MultinomialConjugate, p::ProbabilityParameter)
	# First evaluate in natural co-ordinates
	θ = NaturalParameter(p)
	log_posterior_density(θ) + logabsdetjac(θ)
end
