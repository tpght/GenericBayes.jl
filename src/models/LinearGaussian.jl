export LinearGaussian, GaussianParam
using Distributions
import Distributions.loglikelihood
import Base.map

struct LinearGaussian <: BayesModel
	n_p::Int		# Prior strength
	Σ_p::Matrix{Float64}	# Prior covariance (p x p)
	Σ::Matrix{Float64}	# Noise covariance (k x k)
	μ_p::Vector{Float64}	# Prior mean
	F::Matrix{Float64}	# Forward map
	y::Array{Float64,2}	# k x n data
end

@vector_param GaussianParam

function prior(model::LinearGaussian, ::Type{GaussianParam})
	MvNormal(model.μ_p, (1.0 / model.n_p) * model.Σ_p)
end

function likelihood(model::LinearGaussian, θ::GaussianParam)
	mean = model.F * θ.components
	MvNormal(mean, model.Σ)
end

function loglikelihood(model::LinearGaussian, θ::GaussianParam)
	dist = likelihood(model, θ)
	loglikelihood(dist, model.y)
end

log_posterior_density(model::LinearGaussian, θ::GaussianParam) = logpdf(prior(model, GaussianParam), θ.components) + loglikelihood(model, θ)
prior_mode(model::LinearGaussian, ::Type{GaussianParam}) = GaussianParam(model.μ_p)
sample_size(model::LinearGaussian) = size(model.y)[2]
mle(model::LinearGaussian, ::GaussianParam) = GaussianParam(mean(model.y, dims=2)[:,1])
mle(model::LinearGaussian, ::Type{GaussianParam}) = GaussianParam(mean(model.y, dims=2)[:,1])

function map(model::LinearGaussian, ::GaussianParam)
	Q = inv(model.Σ)
	Q_p = inv(model.Σ_p)
	n = sample_size(model)
	n_p = model.n_p
	t = mle(model, GaussianParam)
	dual = n .* Q * t.components + n_p .* Q_p * model.μ_p
	Σ_post = inv(n*Q + n_p * Q_p)

	GaussianParam((Σ_post * dual)[:,1])
end
