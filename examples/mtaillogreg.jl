using GenericBayes, Distributions, LinearAlgebra
using StatsFuns, Plots, MCMCChains, ForwardDiff, Random, StatProfilerHTML

function make_model(rng, p)
    n = 100                         # Number of observations (data)

    # Generate design matrix
    X = rand(rng, Normal(), (n, p))
    @show β_true = ones(p)
    μ = StatsFuns.logistic.(X * β_true)
    y = rand(rng, Product(Bernoulli.(μ)))

    # Build model
    ρ = 0.99
    Σ = diagm(ones(p))
    Σ[2,1] = ρ
    Σ[1,2] = ρ
    CanonicalGLM{Bernoulli, Float64}(X, y, zeros(p), Σ)
end

# Create samplers
p = 2
k = 1                           # Dimension of embedded m-flat submanifold
N = 100                        # Number of samples
subsampler = SphericalRandomWalk(0.2)
subsamples = Int(100)
# tail_sampler = MTailRecursiveOrthogonalGibbs(NegativeLogDensity(), k, subsampler, subsamples)
it_sampler = EIterativeOrthogonalGibbs(NegativeLogDensity(), k, subsampler, subsamples)
sampler = MRecursiveOrthogonalGibbs(NegativeLogDensity(), k, subsampler, subsamples)

rng = MersenneTwister(1234)
model = make_model(rng, p)

rng = MersenneTwister(1234)
samples = sample(rng, model, sampler, N, chain_type=MCMCChains.Chains);
# @time samples = sample(rng, model, sampler, N, chain_type=MCMCChains.Chains);
# @profilehtml samples = sample(rng, model, sampler, N, chain_type=MCMCChains.Chains);

# rng = MersenneTwister(1234)
# @time tail_samples = sample(rng, model, tail_sampler, N, chain_type=MCMCChains.Chains);

rng = MersenneTwister(1234)
@time it_samples = sample(rng, model, it_sampler, N, chain_type=MCMCChains.Chains);
Chains(it_samples)

@show it_samples == samples
@show it_samples[10], samples[10]
@show maximum(norm(it_samples - samples))

if(p == 2)
    plot(model, it_samples, 1.0)
end

# Is E-recursive on the reversed model the same as m-recursive on the original?
# (after switching samples back, that is)
l = p - k
esampler = ERecursiveOrthogonalGibbs(NegativeLogDensity(), l, subsampler, subsamples)
rng = MersenneTwister(1234)
esamples = sample(rng, model, esampler, N, chain_type=MCMCChains.Chains);

@show samples[50], it_samples[50], esamples[50]


if(p == 2)
    plot(model, esamples, 1.0)
end
