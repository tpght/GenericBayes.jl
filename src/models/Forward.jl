export forward, logprior

function forward(model::BayesModel, θ::Parameter) end

function log_prior_density(model::BayesModel, θ::Parameter)
    check_param(model, θ)
    logpdf(prior(model), Vector(θ))
end
