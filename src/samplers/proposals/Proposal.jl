export Proposal, draw_proposal, log_proposal_ratio
abstract type Proposal end

"""
    draw_proposal

Samples a proposal given a vector θ0, which should have the same dimension as
the target.
"""
function draw_proposal(proposal::Proposal, θ0::Vector{T}) where T<:Real end

"""
     log_proposal_ratio

Computes the logarithm of the proposal probability density ratio of θ1 given θ0,
used in the Metropolis-Hastings accept / reject step
"""
function log_proposal_ratio(proposal::Proposal, θ1::Vector{T},
                              θ0::Vector{T}) where T<:Real end
