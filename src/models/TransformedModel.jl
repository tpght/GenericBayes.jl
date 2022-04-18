export TransformedModel

struct TransformedModel <: BayesModel
    model::BayesModel
    inverse_transform::Function
    logabsdet_d_inverse_transform::Function
end

function log_posterior_density(model::TransformedModel, θ::Vector)
    log_posterior_density(model.model, model.inverse_transform(θ)) +
       model.logabsdet_d_inverse_transform(θ)
end

dimension(model::TransformedModel) = dimension(model.model)
