using DifferentialEquations, Distributions, Optim, ForwardDiff
using DiffEqBase

export ODEModel, ODEParameter

struct ODEModel{S<: DiffEqBase.AbstractODEAlgorithm, T<:Real, F<:Function, DistrType<:Distribution, DataType<:Array} <: BayesModel
    # TODO: Just remake an ODE problem
    problem::ODEProblem         # Contains information about the ODE
    solver::S                   # ODE Solver to use
    times::Vector{T}      # Times at which solution is observed
    observe::F            # Maps solution of problem to a Distribution
    data::DataType        # observations
    prior::DistrType      # prior distribution
    lower::Vector{Float64}      # lower bounds for minimization
    upper::Vector{Float64}      # upper bounds for minimization
end

# Friendly constructor that can take a range
ODEModel(problem::ODEProblem, solver::DiffEqBase.AbstractODEAlgorithm, times::AbstractRange, observe::Function, data::Vector, prior::Distribution, lower::Vector, upper::Vector) = ODEModel(problem, solver, Vector(times), observe, data, prior, lower, upper)

prior(model::ODEModel) = model.prior

@vector_param ODEParameter

function forward(model::ODEModel, θ::ODEParameter)
    check_param(model, θ)

    # Remake the ODE problem at the new parameter
    # Also convert u0 to the same data type as parameter
    # (this allows support for autodiff)
    p = Vector(θ)
    problem = remake(model.problem, p=p, u0=convert.(eltype(p), model.problem.u0))
    solve(problem, model.solver, saveat=model.times)
end

function simulate(model::ODEModel, θ::ODEParameter)
    sol=forward(model::ODEModel, θ::ODEParameter)
    rand(model.observe(sol))
end

function loglikelihood(model::ODEModel, θ::ODEParameter, data::Array)
    # Check that data is the correct size / shape
    # Note: what if model isn't init'd properly?
    size(data) != size(model.data) ? throw(ArgumentError("data passed to loglikelihood is not of shape " * size(model.data))) : nothing
    check_param(model, θ)

    sol = forward(model, θ)     # Solve the ODE for parameters θ
    logpdf(model.observe(sol), data)
end
loglikelihood(model::ODEModel, θ::ODEParameter) = loglikelihood(model, θ, model.data)

function log_posterior_density(model::ODEModel, θ::ODEParameter, data::Array)
    loglikelihood(model, θ, data) + log_prior_density(model, θ)
end
log_posterior_density(model::ODEModel, θ::ODEParameter) = log_posterior_density(model, θ, model.data)

dimension(model::ODEModel) = length(model.problem.p)
lower_box(model::ODEModel, ::Type{ODEParameter}) = model.lower
upper_box(model::ODEModel, ::Type{ODEParameter}) = model.upper
