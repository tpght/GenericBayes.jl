using LinearAlgebra, ForwardDiff, Optim
export Bregman, DualParameter, NegativeLogDensity, Euclidean, Quadratic
export legendre_dual, bregman_generator, inverse_legendre_dual, divergence

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
    divergence

Returns the divergence from θ1 to θ2.
"""
function divergence(θ1, θ2, geometry::Bregman, model::BayesModel)
    Fθ1 = bregman_generator(θ1, geometry, model)
    Fθ2 = bregman_generator(θ2, geometry, model)
    ∇Fθ2 = legendre_dual(θ2, geometry, model)
    Fθ1 - Fθ2 - ∇Fθ2' * (θ1 - θ2)
end

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

function legendre_dual(θ, geometry::NegativeLogDensity, model::BayesModel)
    -grad_log_posterior_density(model, θ)
end

function legendre_dual(θ::Vector{T}, geometry::NegativeLogDensity,
                       model::BayesModel, A::Matrix{T}) where T<:Real
    # Should return A' * ∇F
    -grad_log_posterior_density(model, θ, A)
end

function legendre_dual(θ, geometry::NegativeLogDensity, model::BayesModel, k::Int)
    ηk = -grad_log_posterior_density(model, θ, k)
    [ηk; θ[(k+1):end]]
end

function metric(θ, geometry::NegativeLogDensity, model::BayesModel)
    -hessian_log_posterior_density(model, θ)
end

function metric(θ, geometry::NegativeLogDensity, model::BayesModel, k::Int)
    -hessian_log_posterior_density(model, θ, k)
end

function metric(θ::Vector{T}, geometry::NegativeLogDensity, model::BayesModel,
                A::Matrix{T}) where T<:Real
    # Should return A' * ∇² F * A
    -hessian_log_posterior_density(model, θ, A)
end

"""
    metric(θ, geometry::NegativeLogDensity, model::BayesModel, k::Int, block::Int)

Evaluate the k × k sub-matrix at position block on the block-diagonal of the metric
"""
function metric(θ, geometry::NegativeLogDensity, model::BayesModel, k::Int, block::Int)
    p = length(θ)
    block_inds = ((block-1)*k+1):(block*k)
    upper_inds = (block*k+1):p
    lower_inds = 1:((block-1)*k)

    ForwardDiff.hessian(x -> bregman_generator(
                                                [θ[lower_inds]; x; θ[upper_inds]],
                                                geometry, model),
                        θ[block_inds])
end

function metric_lower(θ, geometry::NegativeLogDensity, model::BayesModel,
                      k::Int)
    # Return lower-right (p-k) × (p-k) block
    ForwardDiff.hessian(x -> -logπ(model, [θ[1:k]; x]), θ[(k+1):end])
end

"""
    legendre_dual(θ::Vector, geometry::Bregman, model::BayesModel)

Compute the Legendre dual, i.e. the gradient of `bregman_generator`.

Default uses automatic differentiation.
Optional parameter k returns mixed co-ordinates (η, θ), where η are the first k
dual co-ordinates.
"""
function legendre_dual(θ, geometry::Bregman, model::BayesModel, k::Int)
    ηk = legendre_dual(θ, geometry, model)[1:k]
    [ηk; θ[(k+1):end]]
end

function legendre_dual(θ::Vector{T}, geometry::Bregman,
                       model::BayesModel, A::Matrix{T}) where T<:Real
    # Should return A' * ∇F
    η = legendre_dual(θ, geometry, model)
    A' * η
end

"""
    inverse_legendre_dual(η, geometry::G, model::BayesModel)

Compute the Legendre dual to η, i.e. the primal vector θ such that
legendre_dual(θ, geometry, model) = η.

Default uses automatic differentiation.
Optional parameter k returns primal co-ordinates corresponding to mixed
co-ordinates (η, θ), where η are the first k dual co-ordinates.
"""
function inverse_legendre_dual(ξ::Vector{T}, geometry::Bregman,
                               model::BayesModel) where T <: Real
    inverse_legendre_dual(ξ, geometry, model, length(ξ))
end

function inverse_legendre_dual(ξ::Vector{T}, geometry::Bregman, model::BayesModel,
                               k::Int; x0=nothing) where T<:Real
    @assert dimension(model) == length(ξ) "Model dimension does not match input"
    if(k == 0) return ξ end

    # @show k
    # @show ξ

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
    proxy(x) = bregman_generator(full_primal(x), geometry, model) - x' * η_k

    function g!(gradient, x)
        gradient .= legendre_dual(full_primal(x), geometry, model, k)[1:k] - η_k
    end

    function h!(hessian, x)
        hessian .= metric(full_primal(x), geometry, model, k)
    end

    # Optimize the function
    # Set a very low tolerance on the gradient
    # TODO add gradient tolerance
    #
    # Newton converges faster, but is more expensive (requires hessian)
    method = Newton()
    # method = ConjugateGradient()
    # Add a gradient tolerance as an optional argument; may increase performance.
    # g_tol = 1e-8
    # result = optimize(proxy, x0, method=method; g_tol=g_tol)
    result = optimize(proxy, g!, h!, x0, method=method)

    if(Optim.converged(result) == false)
        @show result
        @show ξ
        @show x0
        @show k
        # @show k
        error("Could not convert from mixed to primal co-ordinates")
    end

    θ = Optim.minimizer(result)
    [θ; primal]
end

"""
    inverse_legendre_dual

# Arguments
- `δ`: Fixed k-dimensional generalized dual component
- `b`:
- `A`: Matrix whose k columns...
- `geometry`
- `model`
"""
function inverse_legendre_dual(δ::Vector{T}, b::Vector{T}, A::Matrix{T},
 geometry::Bregman, model::BayesModel; x0=nothing) where T<:Real
    @assert dimension(model) == length(b) "Model dimension does not match input"
    k = length(δ)
    if(k == 0) return b end

    @assert size(A)[2] == k "Number of columns in A does not match
                                    length of δ"

    if(x0 == nothing)
        x0 = ones(T, k)
    else
        # Seemingly have to do this to convert to type T (autodiff types)
        x0 = zeros(T,k) + x0
    end

    # Define cost function
    # For conjugate gradient, A is an orthogonal matrix. Therefore, A' A is the
    # k × k identity.
    # Same is true for Gibbs.
    # Calculate projection b of θ onto (Im A)^⟂ = Ker(A')
    full_primal(α) = A * α + b
    proxy(α) = bregman_generator(full_primal(α), geometry, model) - α' * A' * A * δ

    function g!(gradient, α)
        gradient .= legendre_dual(full_primal(α), geometry, model, A) - A' * A * δ
    end

    function h!(hessian, α)
        # NOTE for performance reasons, could use aoto-diff to evaluate this...
        hessian .= metric(full_primal(α), geometry, model, A)
    end

    # Optimize the function
    # Set a very low tolerance on the gradient
    # TODO add gradient tolerance
    method = Newton()
    result = optimize(proxy, g!, h!, x0, method=method, g_tol=1e-10)

    # @show Optim.iterations(result)

    if(Optim.converged(result) == false)
        @show result
        @show b
        @show δ
        @show x0
        error("Could not convert from mixed to primal co-ordinates")
    end

    α = Optim.minimizer(result)
    full_primal(α)
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
function metric(θ, geometry::Bregman, model::BayesModel) end

function metric(θ::Vector{T}, geometry::Bregman, model::BayesModel,
    A::Matrix{T}) where T<:Real
    @assert dimension(model) == length(θ) "Model dimension does not match input"
    G = metric(θ, geometry, model)
    A' * G * A
end

function metric(θ, geometry::Bregman, model::BayesModel, k::Int)
    @assert dimension(model) == length(θ) "Model dimension does not match input"
    # For now, literally evaluate the entire metric and take the upper block
    # TODO maybe print a warning here that this is slow...
    metric(θ, geometry, model)[1:k, 1:k]
end

function metric_lower(θ, geometry::Bregman, model::BayesModel, k::Int)
    @assert dimension(model) == length(θ) "Model dimension does not match input"
    # Evaluate the entire metric and take the lower-right block
    metric(θ, geometry, model)[(k+1):end, (k+1):end]
end

function metric_upperright(θ, geometry::Bregman, model::BayesModel, k::Int)
    @assert dimension(model) == length(θ) "Model dimension does not match input"
    # Evaluate the entire metric and take the lower-right block
    metric(θ, geometry, model)[1:k, (k+1):end]
end

metric(θ, generator::Function) = ForwardDiff.hessian(generator, θ)
metric(θ, generator::Function, k::Int) = ForwardDiff.hessian( x-> generator([x; θ[(k+1):end]]), θ[1:k])
legendre_dual(θ, generator::Function) = ForwardDiff.gradient(generator, θ)
function legendre_dual(θ, generator::Function, k::Int)
    proxy(x) = generator([x; θ[(k+1):end]])
    [ForwardDiff.gradient(proxy, θ[1:k]); θ[(k+1):end]]
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

function inverse_legendre_dual(η::Vector{T}, generator::Function; x0=nothing) where T<:Real
    inverse_legendre_dual(η, generator, length(η), x0=x0)
end

# Returns logabsdet of upper-left k × k block of the metric
function logabsdetmetric(θ, generator::Function, k::Int)
    proxy(x) = generator([x; θ[(k+1):end]])
    logabsdet(ForwardDiff.hessian(proxy, θ[1:k]))[1]
end

logabsdetmetric(θ, generator::Function) = logabsdet(metric(θ, generator))[1]


function logabsdetmetric(θ::Vector{T}, geometry::Bregman, model::BayesModel,
                         A::Matrix{T}) where T<:Real
    G = metric(θ, geometry, model, A)
    if(G isa Float64)
        # This shouldn't happen at all, but hey...
        return log(G)
    end
    logabsdet(metric(θ, geometry, model, A))[1]
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
    return logabsdet(metric(θ, geometry, model))[1]
end

function logabsdetmetric(θ, geometry::Bregman, model::BayesModel, k::Int)
    return logabsdet(metric(θ, geometry, model, k))[1]
end

"""
   logabsdetmetric(θ, geometry::Bregman, model::BayesModel, k::Int, block::Int)

Compute the logabsdet of the k × k sub-matrix on the diagonal of the metric,
at position block
"""
function logabsdetmetric(θ, geometry::Bregman, model::BayesModel, k::Int, block::Int)
    return logabsdet(metric(θ, geometry, model, k, block))[1]
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
"""
struct Euclidean<:Bregman end

bregman_generator(θ, geometry::Euclidean, model::BayesModel) = 0.5 * θ' * θ
metric(θ, geometry::Euclidean, model::BayesModel) = diagm(ones(dimension(model)))
legendre_dual(θ, geometry::Euclidean, model::BayesModel) = θ
legendre_dual(θ, geometry::Euclidean, model::BayesModel, k::Int) = θ
legendre_dual(θ, geometry::Euclidean, model::BayesModel, A::Matrix) = θ
inverse_legendre_dual(θ, geometry::Euclidean, model::BayesModel) = θ
inverse_legendre_dual(θ, geometry::Euclidean, model::BayesModel, k::Int) = θ
inverse_legendre_dual(θ, geometry::Euclidean, model::BayesModel, A::Matrix) = θ

logabsdetmetric(θ, geometry::Euclidean, model::BayesModel, k::Int) = 0.0
logabsdetmetric(θ, geometry::Euclidean, model::BayesModel) = 0.0
logabsdetmetric(θ::Vector{T}, geometry::Euclidean, model::BayesModel,
                A::Matrix{T}) where T<:Real = 0.0


"""
    Quadratic

Dually flat Quadratic geometry; generator of the form 1/2 x^T Λ x - w^T x
"""
struct Quadratic{T<:Real, P}<:Bregman
    Λ::AbstractArray{T, P}
    w::Vector{T}
end

bregman_generator(θ, geometry::Quadratic, model::BayesModel) = 0.5 * θ' * model.Λ * θ - w' * θ
metric(θ, geometry::Quadratic, model::BayesModel) = model.Λ
legendre_dual(θ, geometry::Quadratic, model::BayesModel) = model.Λ * θ - w
inverse_legendre_dual(θ, geometry::Quadratic, model::BayesModel) = model.Λ \ (θ + w)

logabsdetmetric(θ, geometry::Quadratic, model::BayesModel, k::Int) = 0.0
logabsdetmetric(θ, geometry::Quadratic, model::BayesModel) = 0.0

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
