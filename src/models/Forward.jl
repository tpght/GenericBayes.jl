export forward, logprior

function forward(model::BayesModel, θ::Parameter) end

log_prior_density(model::BayesModel, θ::Parameter) = logpdf(prior(model), Vector(θ))

include("ODE.jl")
