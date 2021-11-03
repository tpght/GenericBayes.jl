export MFlatGibbs
"""
    InheritedBregman

Represents geometry on a submanifold inherited from an ambient
dually-flat manifold.
"""
abstract type InheritedBregman <: Bregman end
abstract type InheritedBregmanMFlat <: InheritedBregman end
abstract type InheritedBregmanEFlat <: InheritedBregman end

"""
    MFlatGibbs

Geometry on the (p-k)-dimensional leaves of the m-foliation.
"""
struct MFlatGibbs{T<:Real} <: InheritedBregmanMFlat
    ambient::Bregman            # Geometry on ambient space
    ηk::Vector{T}               # FIXED vector of dual components η1,..,ηk
end

function bregman_generator(θ, geometry::MFlatGibbs, model::MFlatConditional)
    k = length(geometry.ηk)

    # embed into the ambient space
    embed = inverse_legendre_dual([geometry.ηk; θ], geometry.ambient,
                                  ambient_model(model), k)

    F = bregman_generator(embed, geometry.ambient, ambient_model(model))

    # Get the first k components of the embedding
    θk = embed[1:k]

    F - geometry.ηk' * θk
end

function legendre_dual(θ, geometry::MFlatGibbs, model::MFlatConditional)
    k = length(geometry.ηk)

    # embed into the ambient space
    embed = inverse_legendre_dual([geometry.ηk; θ], geometry.ambient,
                                  ambient_model(model), k)

    # Return last p-k components of legendre dual
    legendre_dual(embed, geometry.ambient, ambient_model(model))[(k+1):end]
end

function metric(θ, geometry::MFlatGibbs, model::MFlatConditional)
    k = length(geometry.ηk)

    # embed into the ambient space
    embed = inverse_legendre_dual([geometry.ηk; θ], geometry.ambient,
                                  ambient_model(model), k)

    # Evaluate lower-right (p-k) × (p-k) block of the hessian
    metric_lower(embed, geometry.ambient, ambient_model(model), k)
end

"""
    EFlatGibbs

Geometry on the k-dimensional leaves of the e-foliation.
"""
struct EFlatGibbs{T<:Real} <: InheritedBregmanEFlat
    ambient::Bregman            # Geometry on ambient space
    θok::Vector{T}              # FIXED vector of complementary primal
                                # components θ_{k+1},..,θ_p
end

function bregman_generator(θ, geometry::EFlatGibbs, model::EFlatConditional)
    bregman_generator([θ; geometry.θok], geometry.ambient, ambient_model(model))
end

function legendre_dual(θ, geometry::EFlatGibbs, model::EFlatConditional)
    # Return first k components of legendre dual
    k = length(θ)
    ξ = legendre_dual([θ; geometry.θok],geometry.ambient,ambient_model(model), k)
    ξ[1:k]
end

function metric(θ, geometry::EFlatGibbs, model::EFlatConditional)
    k = length(θ)
    metric([θ; geometry.θok], geometry.ambient, ambient_model(model), k)
end
