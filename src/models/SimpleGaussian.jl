export SimpleGaussian

struct SimpleGaussian{T} <: BayesModel where T<:Real
    ρ::T
end

log_posterior_density(model::SimpleGaussian, θ) = -0.5 * θ' * inv([1.0 model.ρ; model.ρ 1.0]) * θ
dimension(model::SimpleGaussian) = 2
