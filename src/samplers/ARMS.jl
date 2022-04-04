import AbstractMCMC.AbstractSampler, AbstractMCMC.step
export ARMS, set_initial

"""
    ARMS

Random walk, Gaussian proposal with a spherical covariance.
"""
struct ARMS{T<:Real} <: AbstractSampler
    a::Vector{T}                        # Left endpoint
    b::Vector{T}                        # Right endpoint
    width::T
    metropolis::Bool            # Use metropolis accept/reject

    # Constructor
    function ARMS(width::T, metropolis::Bool) where T<:Real
        # Load the ARMSpp library with RCall.jl
        # @rlibrary armspp
        R"library(armspp)"
        a = [-width]
        b = [width]
        new{T}(a,b,width,metropolis)
    end
end

function set_initial(sampler::ARMS{T}, v::Vector{T}) where T<:Real
        sampler.a .= [v[1]-sampler.width]
        sampler.b .= [v[1]+sampler.width]
end

"""
    sample(rng, model::BayesModel, sampler::ARMS{G}; kwargs...)

Sample from model using ARMS
"""
function sample(model::BayesModel, sampler::ARMS, N::Integer;
                kwargs...)
    GenericBayes.sample(Random.GLOBAL_RNG, model, sampler, N; kwargs...)
end
function sample(rng::AbstractRNG, model::BayesModel,
                sampler::ARMS, N::Integer;
                kwargs...)

    # TODO pass rng to R
    start=time()

    log_density(x) = log_posterior_density(model, [x])
    # log_density(x) = -x^2 /2

    # samples = rcopy(R"arms($N, $log_density, $sampler.a, $sampler.b, metropolis = $sampler.metropolis)")
    #R"samples<-arms($N, $log_density, $sampler.a, $sampler.b, metropolis=$sampler.metropolis)"
    samples = rcopy(rcall(:arms, N, log_density, sampler.a[1], sampler.b[1], metropolis=sampler.metropolis))

    if(N==1)
        samples = [[samples]]
    else
        samples = Vector([[sample] for sample in samples])
    end

    # @rget samples
    # samples = @rget output

    stop=time()
    times = (start=start,stop=stop)
    return bundle_samples(samples, model, sampler, samples[end],
                          MCMCChains.Chains, stats=times)
end
