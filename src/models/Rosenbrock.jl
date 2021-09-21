export Rosenbrock, RosenbrockParameter

"""
    Rosenbrock{T<:Real}<:BayesModel

The probability distribution whose negative log density is the Rosenbrock function
"""
struct Rosenbrock{T<:Real}<:BayesModel
    a::T
    b::T
end

Rosenbrock() = Rosenbrock{Float64}(1.0, 100.0)

@vector_param RosenbrockParameter

function log_posterior_density(model::Rosenbrock, θ::RosenbrockParameter)
    -(model.a - θ.components[1])^2 - model.b * (θ.components[2] - θ.components[1]^2)^2
end

dimension(model::Rosenbrock) = 2 # For now

function max_posterior(model::Rosenbrock, θ::RosenbrockParameter)
    RosenbrockParameter([model.a, model.a^2])
end
