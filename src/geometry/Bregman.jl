using LinearAlgebra, ForwardDiff, Optim
export Bregman, DualParameter, NegativeLogDensity, Euclidean
export legendre_dual, bregman_generator, inverse_legendre_dual

"""
    Bregman

Geometry derived from a Bregman divergence.
"""
abstract type Bregman <: Geometry end

"""
    bregman_generator

The convex function from which the Bregman divergence and geometry are derived
"""
function bregman_generator(θ, geometry::Bregman, model::BayesModel) end

"""
    NegativeLogDensity{P}

Bregman geometry which uses the entire negative log posterior density as the generator.

Of course, the posterior density depends on parameterization. Therefore this
geometry has P<:Parameter to indicate which co-ordinates defines the geometry.
"""
struct NegativeLogDensity<:Bregman end

function bregman_generator(θ, geometry::NegativeLogDensity, model::BayesModel)
    return -log_posterior_density(model, θ)
end

"""
    legendre_dual(θ::Parameter, geometry::Bregman, model::BayesModel)

Compute the Legendre dual, i.e. the gradient of `bregman_generator`.

Default uses automatic differentiation.
Optional parameter k returns mixed co-ordinates (η, θ), where η are the first k
dual co-ordinates.
"""
function legendre_dual(θ, geometry::Bregman, model::BayesModel)
    # First compute the gradient as a vector
    proxy(x) = bregman_generator(x, geometry, model)
    ForwardDiff.gradient(proxy, θ)
end

function legendre_dual(θ, geometry::Bregman, model::BayesModel, k::Int)
    p = length(θ)
    primal = θ[(k+1):p]
    # Differentiate
    proxy(x) = bregman_generator([x; primal], geometry, model)
    η = ForwardDiff.gradient(proxy, θ[1:k])
    [η; primal]
end

"""
    inverse_legendre_dual(η, geometry::G, model::BayesModel)

Compute the Legendre dual to η, i.e. the primal vector θ such that
legendre_dual(θ, geometry, model) = η.

Default uses automatic differentiation.
Optional parameter k returns primal co-ordinates corresponding to mixed
co-ordinates (η, θ), where η are the first k dual co-ordinates.
"""
function inverse_legendre_dual(η::Vector{T}, geometry::G,
                               model::BayesModel) where G<:Bregman where T<:Real

    # TODO Find an appropriate initial point x0
    x0 = ones(T, size(η, 1))

    # Define cost function
    proxy(x) = bregman_generator(x, geometry, model) - x' * η

    # Optimize the function
    # lower = lower_box(model, P)
    # upper = upper_box(model, P)
    result = optimize(proxy, x0, LBFGS(); autodiff=:forward)

    if(Optim.converged(result) == false)
        @show Optim.converged(result)
        @show Optim.iterations(result)
        @show Optim.iteration_limit_reached(result)
        error("Could not convert from dual to primal co-ordinates")
    end

    return Optim.minimizer(result)
end

# function inverse_legendre_dual(ξ::Vector{T}, geometry::G,
#                                model::BayesModel, k::Int) where G<:Bregman where T<:Real

#     # TODO Find an appropriate initial point x0
#     # TODO Fix this function! Use ConstrainedOptim.jl
#     p = length(ξ)
#     primal = ξ[(k+1):p]
#     dual = ξ[1:k]

#     # Define cost function
#     full(x) = [x; primal]
#     proxy(x) = bregman_generator([x; primal], geometry, model)
#     ∇F(x) = ForwardDiff.gradient(proxy, x) # First k dual components
#     ℒ(x, λ) = bregman_generator(full(x), geometry, model) -
#         x' * dual -
#         primal' * ∇F(x) -
#         sum(λ .* (∇F(x) - dual))

#     function ℒ(L, z)
#         ℒ(z[1:k], z[(k+1):(2*k)])
#     end

#     # Optimize the function
#     # lower = lower_box(model, P)
#     # upper = upper_box(model, P)
#     result = nlsolve(ℒ(z), ones(T, 2*k), LBFGS(); autodiff=:forward)

#     if(Optim.converged(result) == false)
#         @show Optim.converged(result)
#         @show Optim.iterations(result)
#         @show Optim.iteration_limit_reached(result)
#         error("Could not convert from mixed to primal co-ordinates")
#     end

#     θ = Optim.minimizer(result)
#     [θ; primal]
# end

"""
    dual_bregman_generator(θ, geometry, model)

Evaluate the Legendre dual (convex conjugate) to `bregman_generator` at θ
"""
function dual_bregman_generator(θ::Vector{T}, geometry::Bregman,
                                model::BayesModel) where T<:Real
    η = legendre_dual(θ, geometry, model)
    return θ' * η - bregman_generator(θ, geometry, model)
end

"""
    metric(θ, geometry::Bregman, model::BayesModel)

Compute the Riemannian metric, i.e. the hessian of `bregman_generator`.
"""
function metric(θ, geometry::Bregman, model::BayesModel) where T<:Real
    # Default uses autodiff
    proxy(x) = bregman_generator(x, geometry, model)
    ForwardDiff.hessian(proxy, θ)
end

# """
#     metric(η::DualParameter{T, G, P}, geometry::Bregman, model::BayesModel)

# Compute the Riemannian metric, i.e. the inverse hessian of `bregman_generator`.
# """
# function metric(η::DualParameter{T, G, P}, geometry::G,
#                        model::BayesModel) where G<:Bregman where
# P<:Parameter{T} where T<:Real
#     # Default uses autodiff
#     ParameterType = Base.typename(P).wrapper

#     # For now, literally just do the inversion
#     θ = legendre_dual(η, geometry, model)
#     my_metric = metric(θ, geometry, model)

#     inv(my_metric)
# end

"""
    logabsdetmetric(θ, geometry::Bregman, model::BayesModel)

Compute the log absolute-value of the determinant of the Riemannian metric
"""
function logabsdetmetric(θ, geometry::Bregman, model::BayesModel)
    return log(abs(det(metric(θ, geometry, model))))
end

# """
#     grad_logabsdetmetric(θ, geometry::Bregman, model::BayesModel)

# Compute the gradient of logabsdet of the Riemannian metric
# """
# function grad_logabsdetmetric(θ, geometry::Bregman, model::BayesModel) where
# T<:Real

#     PrimalType = Base.typename(typeof(θ)).wrapper
#     proxy(x) = logabsdetmetric(PrimalType(x), geometry, model)
#     return ForwardDiff.gradient(proxy, θ.components)
# end

"""
    Euclidean

Dually flat Euclidean geometry. Primal and dual co-ordinates are identical.

The Parameter P indicates the co-ordinate system in which the Riemannian metric
is the identity.
"""
struct Euclidean<:Bregman end

function bregman_generator(θ, geometry::Euclidean, model::BayesModel)
   return 0.5 * θ' * θ
end

function legendre_dual(θ, geometry::Euclidean, model::BayesModel)
    return θ
end

# function metric(θ::P, geometry::Euclidean{P}, model::BayesModel) where
#     P<:Parameter{T} where T<:Real
#     return diagm(ones(dimension(model)))
# end

# function metric(η::DualParameter{T, G, P}, geometry::G,
#                        model::BayesModel) where
# P<:Parameter{T} where T<:Real where G<:Euclidean
#     return diagm(ones(dimension(model)))
# end

# function legendre_dual(η::DualParameter{T, G, P}, geometry::G,
#                        model::BayesModel) where
# P<:Parameter{T} where T<:Real where G<:Euclidean
#     return P(η.components)
# end

# function dual_bregman_generator(θ, geometry::Euclidean,
#                                 model::BayesModel) where T<:Real
#     return 0.5 * θ.components' * θ.components
# end

# function logabsdetmetric(θ, geometry::Euclidean, model::BayesModel) where
# T<:Real
#     return 0.0
# end

# function grad_logabsdetmetric(θ, geometry::Euclidean, model::BayesModel) where
# T<:Real

#     return zeros(dimension(model))
# end
