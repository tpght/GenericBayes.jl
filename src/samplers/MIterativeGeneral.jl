import AbstractMCMC.AbstractSampler, AbstractMCMC.step
export MIterativeGeneral, MOrthogonalGradient, GeneralMGibbs

"""
    MIterativeGeneral

Orthogonal Gibbs sampling algorithm: generalizes e/m-recursive versions
"""
abstract type MIterativeGeneral <: AbstractSampler end

"""
    block_size

The dimension of the embedded submanifolds which are sampled from each
iteration.
"""
function block_size(sampler::MIterativeGeneral) end

"""
    block_basis

Compute new vectors for the A-matrix. For example, orthogonal gibbs should
return standard basis vectors; orthogonal gradient returns gradient.
"""
function block_basis!(A::Matrix{T}, θ::Vector{T}, model::BayesModel,
                     sampler::MIterativeGeneral,
                     block::Int) where T<:Real end

"""
    geometry

Returns the geometry for a given iterative general sampler.
"""
geometry(sampler::MIterativeGeneral) = sampler.geometry

"""
    subsampler

Returns the subsampler for a given iterative general sampler.
"""
function subsampler(sampler::MIterativeGeneral)::AbstractSampler end

"""
    subsamples

Returns the subsampler for a given iterative general sampler.
"""
function subsamples(sampler::MIterativeGeneral)::Int end

"""
    stop_condition

If TRUE, the sampler will return the latest iterate and terminate.
"""
function stop_condition(sampler::MIterativeGeneral, block::Int, θ)::Bool
    return false
end

struct MOrthogonalGradient{T<:Real} <: MIterativeGeneral
    geometry::Bregman           # Geometry to be used in the sampler
    subsampler::AbstractSampler # Sampler to be used on each e-flat submanifold
    subsamples::Int                  # Number of times to run the embedded sampler
    initial_θ::Vector{T}
    stop::Int                   # Stop after this many steps
end

"""
    stop_condition

The MOrthogonalGradient method will terminate after block=stop steps of the
algorithm.
"""
function stop_condition(sampler::MOrthogonalGradient, block::Int, θ)
    return (block >= sampler.stop)
end

function set_initial(sampler::MOrthogonalGradient{T}, v::Vector{T}) where
    T<:Real
    sampler.initial_θ[1:length(v)] .= v
end

function block_basis!(A::Matrix{T}, θ::Vector{T}, model::BayesModel,
                     sampler::MOrthogonalGradient,
                     block::Int) where T<:Real
    gradient = grad_log_posterior_density(model, θ)

    # block size is one with this sampler.
    # Gram-Schmidt with respect to Euclidean inner-product on A-basis
    # The following is a numerically stable Gram-Schmidt
    # Shamelessly taken from https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    A[:, block]=gradient
    # for j=1:(block-1)
    #     A[:,block]=A[:,block]-(A[:,j]'*A[:,block]
    #                 /(norm(A[:,j]))^2) * A[:,j]
    # end
    # A[:, block] = A[:, block]/norm(A[:, block]);
end

block_size(sampler::MOrthogonalGradient) = 1
subsamples(sampler::MOrthogonalGradient) = sampler.subsamples
subsampler(sampler::MOrthogonalGradient) = sampler.subsampler


struct GeneralMGibbs{T<:Real} <: MIterativeGeneral
    geometry::Bregman           # Geometry to be used in the sampler
    block_size::Int
    subsampler::AbstractSampler # Sampler to be used on each e-flat submanifold
    subsamples::Int                  # Number of times to run the embedded sampler
    initial_θ::Vector{T}
end

# Don't need to define block_basis, since default is identity matrix
block_size(sampler::GeneralMGibbs) = sampler.block_size
subsamples(sampler::GeneralMGibbs) = sampler.subsamples
subsampler(sampler::GeneralMGibbs) = sampler.subsampler

"""
    step(rng, model::BayesModel, sampler::MIterativeGeneral,
              state=nothing; kwargs...)

One iteration of a MIterativeGeneral type method.
"""
function step(rng, model::BayesModel, sampler::MIterativeGeneral,
              current_state=nothing; kwargs...) where T<:Real

    p = dimension(model)
    bs = block_size(sampler)
    nblocks = Int(p / bs)

    # First, generate an initial state if required
    if (current_state == nothing)
        state = sampler.initial_θ[1:dimension(model)]
        return state, state
        # η = legendre_dual(θ, geometry(sampler), model, p-k)
        # return θ, [θ, η]
    end

    # Stores the basis of tangent vectors spanning each e-flat submanifold
    A = diagm(ones(p))
    δ = zeros(p)

    block_model = BlockModel(A, δ, copy(current_state), zeros(p),
                         ones(Int, 1),block_size(sampler),
                             sampler.geometry, model)


    for block = 1:nblocks
        # Update block in struct
        block_model.block[1] = block

        # Update the A matrix at the current point
        block_basis!(block_model.A, block_model.θ, model, sampler, block)

        lower_inds = 1:((block-1)*bs)
        block_inds = ((block-1)*bs+1):(block*bs)
        Ap = A[:,lower_inds]

        block_model.δ[lower_inds] .= legendre_dual(block_model.θ,
                                                   geometry(sampler), model, Ap)

        if(block == 1)
            block_model.b[:] = block_model.θ[:] - Ap * pinv(Ap) * block_model.θ[:]
        end

        # Get initial guess from current value of θ
        set_initial(subsampler(sampler), A[:,block_inds]' * block_model.θ)
        subsample = AbstractMCMC.sample(rng,
                                        block_model,
                                        subsampler(sampler),
                                        subsamples(sampler),
                                        progress=false)

        # Save a subsample in the current block

        block_model.θ .= membed(subsample, block_model)
        # embed = membed(subsample, block_model)
        # Ap = A[:,[lower_inds; block_inds]]
        # block_model.θ .= block_model.θ + Ap * pinv(Ap) * (embed - block_model.θ)

        # Check if stop condition is reached
        if(stop_condition(sampler, block, block_model.θ))
            return block_model.θ, block_model.θ
        end
    end

    # The first returned value is the sample, i.e. in primal co-ordinates
    return block_model.θ, block_model.θ
end

struct BlockModel{T<:Real} <: BayesModel
    A::Matrix{T}
    δ::Vector{T}
    θ::Vector{T}
    b::Vector{T}
    block::Vector{Int}
    block_size::Int
    geometry::Bregman
    ambient_model::BayesModel
end

dimension(model::BlockModel) = model.block_size

function membed(x::Vector{T}, model::BlockModel{T}) where T<:Real
    # embed into the ambient space
    # NOTE: β component is where x goes, I think!
    # Project onto (Im A)^⟂, A = Ap here...
    # That gives b := B * β.
    # NOW the component of b in the direction of the ORTHOGONALIZED,
    # GRAM-SCHMIDTED gradient is what needs updating.
    # TODO Can speed this up: Assume A is orthogonal. No need to compute
    # generalized inverse.

    A = model.A
    bs = model.block_size
    block = model.block[1]
    block_inds = ((block-1)*bs+1):(block*bs)
    lower_inds = 1:((block-1)*bs)
    C = model.A[:, block_inds]
    Ap = A[:,lower_inds]
    θ = model.θ

    # NOTE: possibly inefficiency here!
    # Don't have to compute this each time...
    # pinv(A)  = A' if A is ortho

    # Project onto (ImAp)^⟂, where Ap is basis vectors from earlier iterations.
    #
    # TEMP: Do we need this line???
    model.b[:] = θ[:] - Ap * (pinv(Ap) * θ[:])
    # model.b[:] = model.b[:] + C * (x - (C' * model.b[:]))
    model.b[:] = model.b[:] + C * x
    @show norm(Ap' * model.b[:])
    @show norm(Ap' * C)
    x0 = Ap' * θ
    δ = model.δ[lower_inds]

    inverse_legendre_dual(δ, model.b, Ap,
                          model.geometry,
                          model.ambient_model, x0=x0)
end

# Sample block conditionally on values of η filled in thus far, and
# values of θ from the next block onwards.
function log_posterior_density(model::BlockModel{T}, x::Vector{T}) where T<:Real

    # NOTE Here we evaluate a big sub-matrix.
    # what if we simply evaluate the k × k block on the diagonal, in
    # the correct position for the block?
    # A^T G A
    embed = membed(x, model)
    bs = model.block_size
    block = model.block[1]
    lower_inds = 1:((block-1)*bs)
    Ap = model.A[:,lower_inds]

    jac_term = logabsdetmetric(embed, model.geometry,
                               model.ambient_model, Ap)
    GenericBayes.logπ(model.ambient_model, embed) - jac_term
end

function bundle_samples(samples,
    model::BlockModel,
    sampler::AbstractSampler,
    current_state::Any,
    ::Type;
    kwargs...
)
    # ONLY interested in last sample (embedding)
    # TODO Make this the embedding...
    return current_state
end
