export ChenIbrahimGLM

struct ChenIbrahimGLM{T<:Real} <: BayesModel
	n0::T						# Prior sample size
    y0::Vector{T}				# Prior predictions for data
	ψ::Function					# Cumulant generating function
	X::Matrix{T}				# Design matrix
end

dimension(model::ChenIbrahimGLM) = size(model.X)[2]

# Chen-Ibrahim prior for a generalized linear model.
# β are the coefficients of a generalized linear model

function cgf(model::ChenIbrahimGLM, β::Vector{T}) where T<:Real
	sum(model.ψ.(model.X * β))
end

function log_posterior_density(model::ChenIbrahimGLM, β::Vector{T}) where T<:Real
    t = model.X' * model.y0
	model.n0 * (β' * t - cgf(model, β))
end
