using GenericBayes, Distributions, LinearAlgebra
using StatsFuns, Plots, MCMCChains, ForwardDiff, Random

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
p = 3
l = 1                           # Dimension of embedded m-flat submanifold
N = 100                        # Number of samples
subsampler = SphericalRandomWalk(0.5)
subsamples = Int(100)

it_sampler = EIterativeOrthogonalGibbs(NegativeLogDensity(), l, subsampler, subsamples)
itm_sampler = MIterativeOrthogonalGibbs(NegativeLogDensity(), l, subsampler, subsamples)
sampler = ERecursiveOrthogonalGibbs(NegativeLogDensity(), l, subsampler, subsamples)
msampler = MRecursiveOrthogonalGibbs(NegativeLogDensity(), l, subsampler, subsamples)
gsampler = OrthogonalGibbs(NegativeLogDensity(), l, subsampler, subsamples)

rng = MersenneTwister(1234)
model = make_model(rng, p)

rng = MersenneTwister(1234)
@time samples = sample(rng, model, sampler, N, chain_type=MCMCChains.Chains);

rng = MersenneTwister(1234)
@time msamples = sample(rng, model, msampler, N, chain_type=MCMCChains.Chains);

rng = MersenneTwister(1234)
@time it_samples = sample(rng, model, it_sampler, N, chain_type=MCMCChains.Chains);

rng = MersenneTwister(1234)
@time gsamples = sample(rng, model, gsampler, N, chain_type=MCMCChains.Chains);

rng = MersenneTwister(1234)
@time itm_samples = sample(rng, model, itm_sampler, N, chain_type=MCMCChains.Chains);

# Is the tail recursive algorithm equivalent to the usual recursive one?

@show gsamples == it_samples
@show gsamples == itm_samples

@show norm(gsamples[100]- itm_samples[100], 2)

if(p == 2)
    plot(model, tail_samples, 1.0)
end
