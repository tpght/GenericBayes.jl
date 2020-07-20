export sample_size

abstract type DiaconisConjugate<:BayesModel end

function log_posterior_density(model::DiaconisConjugate, θ::NaturalParameter)
	t = sufficient_statistic(model)
	ndata = sample_size(model)
	sz = ndata + prior_sample_size(model)
	λ = ndata / sz
	newt = λ .* t .+ (1 - λ) .* diaconis_hyperparam(model)
	sz * (θ.components' * newt - ψ(model, θ))
end

function grad_log_posterior_density(model::DiaconisConjugate, θ::NaturalParameter)
	ndata = sample_size(model)
	sz = ndata + prior_sample_size(model)
	λ = ndata / sz
	t = sufficient_statistic(model)
	newt = λ .* t .+ (1 - λ) .* diaconis_hyperparam(model)
	sz * (newt - ∇ψ(model, θ))
end

function mle(model::DiaconisConjugate, ::Type{MeanParameter})
	MeanParameter(sufficient_statistic(model))
end

function prior_mode(model::DiaconisConjugate, ::Type{MeanParameter})
	MeanParameter(diaconis_hyperparam(model))
end


include("Multinomial.jl")
