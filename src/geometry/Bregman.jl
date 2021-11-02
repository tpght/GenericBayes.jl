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

legendre_dual(θ, generator::Function) = ForwardDiff.gradient(generator, θ)

function legendre_dual(θ, geometry::Bregman, model::BayesModel, k::Int)
    legendre_dual(θ, x -> bregman_generator(x, geometry, model), k)
end

function legendre_dual(θ, generator::Function, k::Int)
    proxy(x) = generator([x; θ[(k+1):end]])
    [ForwardDiff.gradient(proxy, θ[1:k]); θ[(k+1):end]]
end

"""
    inverse_legendre_dual(η, geometry::G, model::BayesModel)

Compute the Legendre dual to η, i.e. the primal vector θ such that
legendre_dual(θ, geometry, model) = η.

Default uses automatic differentiation.
Optional parameter k returns primal co-ordinates corresponding to mixed
co-ordinates (η, θ), where η are the first k dual co-ordinates.
"""
function inverse_legendre_dual(η::Vector{T}, generator::Function; x0=nothing) where T<:Real
    inverse_legendre_dual(η, generator, length(η), x0=x0)
end

function inverse_legendre_dual(η::Vector{T}, geometry::G,
                               model::BayesModel; x0=x0) where G<:Bregman where T<:Real
    @assert dimension(model) == length(η) "Model dimension does not match input"
    inverse_legendre_dual(η, x->bregman_generator(x, geometry, model); x0=x0)
end

function inverse_legendre_dual(η::Vector{T}, geometry::G,
                               model::BayesModel, k::Int; x0=nothing) where G<:Bregman where T<:Real
    @assert dimension(model) == length(η) "Model dimension does not match input"
    inverse_legendre_dual(η, x->bregman_generator(x, geometry, model), k; x0=x0)
end

function inverse_legendre_dual(ξ::Vector{T}, generator::Function,
                               k::Int; x0=nothing) where T<:Real
    primal = ξ[(k+1):end]
    η_k = ξ[1:k]

    if(x0 == nothing)
        x0 = ones(T, k)
    else
        # Seemingly have to do this to convert to type T (autodiff types)
        x0 = zeros(T,k) + x0
    end

    # Define cost function
    full_primal(x) = [x; primal]
    proxy(x) = generator(full_primal(x)) - x' * η_k

    function g!(gradient, x)
        gradient .= legendre_dual(full_primal(x), generator, k)[1:k] - η_k
    end

    function h!(hessian, x)
        hessian .= metric(full_primal(x), generator, k)
    end

    # Optimize the function
    # Set a very low tolerance on the gradient
    # TODO add gradient tolerance
    result = optimize(proxy, g!, h!, x0, Newton())
    # result = optimize(proxy, g!, x0, ConjugateGradient())
    # result = optimize(proxy, x0, Newton(); autodiff= :forward)

    # method = ConjugateGradient(linesearch=BackTracking())
    # result = optimize(proxy, x0, method=method; autodiff=:forward,
    #                   g_tol=1e-10, x_tol=1e-10, f_tol=1e-10)

    if(Optim.converged(result) == false)
        @show result
        # @show k
        error("Could not convert from mixed to primal co-ordinates")
    end

    θ = Optim.minimizer(result)
    [θ; primal]
end

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
metric(θ, generator::Function) = ForwardDiff.hessian(generator, θ)
metric(θ, generator::Function, k::Int) = ForwardDiff.hessian( x-> generator([x; θ[(k+1):end]]), θ[1:k])
function metric(θ, geometry::Bregman, model::BayesModel)
    @assert dimension(model) == length(η) "Model dimension does not match input"
    metric(θ, x -> bregman_generator(x, geometry, model))
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
logabsdetmetric(θ, generator::Function) = logabsdet(metric(θ, generator))[1]
function logabsdetmetric(θ, geometry::Bregman, model::BayesModel)
    return logabsdet(metric(θ, geometry, model))[1]
end

function logabsdetmetric(θ, geometry::Bregman, model::BayesModel, k::Int)
    logabsdetmetric(θ, x  -> bregman_generator(x, geometry, model), k)
end


# Returns logabsdet of upper-left k × k block of the metric
function logabsdetmetric(θ, generator::Function, k::Int)
    proxy(x) = generator([x; θ[(k+1):end]])
    logabsdet(ForwardDiff.hessian(proxy, θ[1:k]))[1]
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

bregman_generator(θ, geometry::Euclidean, model::BayesModel) = 0.5 * θ' * θ

legendre_dual(θ, geometry::Euclidean, model::BayesModel) = θ
inverse_legendre_dual(θ, geometry::Euclidean, model::BayesModel) = θ

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
