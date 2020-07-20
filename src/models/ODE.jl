using DifferentialEquations, Distributions, Optim, ForwardDiff
using DiffEqBase

export ODEModel, ODEParameter

struct ODEModel{S<: DiffEqBase.AbstractODEAlgorithm, T<:Real, F<:Function, DistrType<:Distribution, DataType<:Array} <: BayesModel
    # TODO: Just remake an ODE problem
    problem::ODEProblem         # Contains information about the ODE
    solver::S                   # ODE Solver to use
    times::Vector{T}      # Times at which solution is observed
    observe::F            # Maps solution of problem to a Distribution
    data::DataType
    prior::DistrType
end

# Friendly constructor that can take a range
ODEModel(problem::ODEProblem, solver::DiffEqBase.AbstractODEAlgorithm, times::AbstractRange, observe::Function, data::Vector, prior::Distribution) = ODEModel(problem, solver, Vector(times), observe, data, prior)

prior(model::ODEModel) = model.prior

@vector_param ODEParameter

function forward(model::ODEModel, θ::ODEParameter)
    p = Vector(θ)

    # Remake the ODE problem at the new parameter
    # Also convert u0 to the same data type as parameter
    # (this allows support for autodiff)
    problem = remake(model.problem, p=p, u0=convert.(eltype(p), model.problem.u0))
    solve(problem, model.solver, saveat=model.times)
end

function simulate(model::ODEModel, θ::ODEParameter)
    sol=forward(model::ODEModel, θ::ODEParameter)
    rand(model.observe(sol))
end

function loglikelihood(model::ODEModel, θ::ODEParameter)
    sol = forward(model, θ)     # Solve the ODE for parameters θ
    logpdf(model.observe(sol), model.data)
end

function log_posterior_density(model::ODEModel, θ::ODEParameter)
    loglikelihood(model, θ) + log_prior_density(model, θ)
end
