export RandomWalkProposal

struct RandomWalkProposal{T<:Real} <: Proposal
    scale::T
end

"""
    draw_proposal

Samples a proposal given a vector θ0, which should have the same dimension as
the target.
"""
function draw_proposal(rng, proposal::RandomWalkProposal{T}, θ0::Vector{T}) where T<:Real
    σ = proposal.scale
    dim = length(θ0)

    # Generate a proposal
    proposal = θ0 .+ rand(rng, Normal(0.0, σ), dim)
end

# Ratio of proposal densities is zero for this proposal.
log_proposal_ratio(proposal::RandomWalkProposal{T}, θ1::Vector{T},
                   θ0::Vector{T}) where T<:Real = 0.0
