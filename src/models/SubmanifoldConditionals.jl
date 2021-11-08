export MFlatConditionalGibbs
"""
    MFlatConditionalGibbs

Conditions a probability distribution onto an m-flat submanifold.
"""
struct MFlatConditionalGibbs{T<:Real} <: MFlatConditional
    ambient_model::BayesModel # Model on the ambient space
    ambient_geometry::Bregman   # Geometry in ambient manifold
    ηk::Vector{T}               # First k dual components
    θk0::Vector{T}              # Initial guess for first k primal components
end

ambient_model(model::MFlatConditionalGibbs) = model.ambient_model
dimension(model::MFlatConditionalGibbs) = dimension(model.ambient_model) - length(model.ηk)

function log_posterior_density(model::MFlatConditionalGibbs, θok)
    @assert length(θok) == dimension(model) "Input size does not equal model dimension"

    # θok is the primal components θ_{k+1},.., θ_p
    k = length(model.ηk)

    # Build the full primal vector by embedding into total space
    θ = inverse_legendre_dual([model.ηk; θok], model.ambient_geometry,
    ambient_model(model), k, x0=model.θk0)

    logπ(ambient_model(model), θ) - logabsdetmetric(θ, model.ambient_geometry,
                                                   ambient_model(model), k)
end


function bundle_samples(
    samples,
    model::SubmanifoldConditional,
    ::AbstractMCMC.AbstractSampler,
    current_state::Any,
    ::Type
)
    # ONLY interested in last sample (embedding)
    # TODO Make this the embedding...
    return current_state
end


"""
    EFlatConditionalGibbs

Conditions a probability distribution onto a k-dimensional e-flat submanifold.
"""
struct EFlatConditionalGibbs{T<:Real} <: EFlatConditional
    ambient_model::BayesModel # Model on the ambient space
    θok::Vector{T}            # Complementary primal components
end

ambient_model(model::EFlatConditionalGibbs) = model.ambient_model

dimension(model::EFlatConditionalGibbs) = dimension(model.ambient_model) - length(model.θok)

function log_posterior_density(model::EFlatConditionalGibbs, θk)
    @assert length(θk) == dimension(model) "Input size does not equal model dimension"
    logπ(model.ambient_model, [θk; model.θok])
end
