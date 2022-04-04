"""
    lower_box(model)

Define the lower bounds for parameterization of `model`.

Default returns `-Inf` in each component, i.e. no lower bounds.
"""
lower_box(model::BayesModel) = [-Inf for i in 1:dimension(model)]

"""
    upper_box(model, P)

Define the upper bounds for parameterization of `model`.

Default returns `Inf` in each component, i.e. no upper bounds.
"""
upper_box(model::BayesModel) = [Inf for i in 1:dimension(model)]

"""
    mle(model, θ, data)

the maximum likelihood estimate (mle) of `model` when parameterized by `typeof(θ)`
and passing `data` to `loglikelihood`.

Optimization uses `θ` as an initial guess.
"""
function mle(model::BayesModel, θ, data::Array)
    cost(x) = -loglikelihood(model, x, data)
    od = OnceDifferentiable(cost, θ; autodiff=:forward)
    lower = lower_box(model)
    upper = upper_box(model)
    Optim.minimizer(optimize(od, lower, upper, θ, Fminbox(BFGS())))
end


function mle(model::BayesModel, θ)
    cost(x) = -loglikelihood(model, x)
    od = OnceDifferentiable(cost, θ; autodiff=:forward)
    lower = lower_box(model)
    upper = upper_box(model)
    Optim.minimizer(optimize(od, lower, upper, θ, Fminbox(BFGS())))
end

"""
    max_posterior(model, θ, data)

The maximum-a-posterior (max_posterior) of `model` when parameterized by `typeof(θ)`
and passing `data` to `loglikelihood`.

Optimization uses `θ` as an initial guess.
"""
function max_posterior(model::BayesModel, θ, data::Array)
    cost(x) = -log_posterior_density(model, x, data)
    od = OnceDifferentiable(cost, θ; autodiff=:forward)
    lower = lower_box(model)
    upper = upper_box(model)
    Optim.minimizer(optimize(od, lower, upper, θ, Fminbox(BFGS())))
end

function max_posterior(model::BayesModel, θ)
    cost(x) = -log_posterior_density(model, x)
    od = OnceDifferentiable(cost, θ; autodiff=:forward)
    lower = lower_box(model)
    upper = upper_box(model)
    Optim.minimizer(optimize(od, lower, upper, θ, Fminbox(BFGS())))
end


# """
#     check_param(model, θ)

# Returns True if `θ` is a permissible parameter for `model`, i.e.
# lies in the parameter space of `model`.
# """
# function check_param(model::BayesModel, θ::Chart)
#     # Check parameter has the right dimension
#     # NOTE: This check might fail unnecessarily;
#     # e.g. a probability parameter has length n+1
#     length(θ) == dimension(model) ? nothing : throw(ArgumentError(
#     "Chart has dimension $(length(θ)); model is dimension $(dimension(θ))"))

#     # Check parameter is in bounds
#     ChartType = Base.typename(typeof(θ)).wrapper
#     p = Vector(θ)
#     all((p .> lower_box(model, ChartType)) .&
#     (p .< upper_box(model, ChartType))) ?
#     nothing : throw(ArgumentError("Chart is out of bounds: $(Vector(θ))"))
# end
