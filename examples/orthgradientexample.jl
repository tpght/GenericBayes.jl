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
p = 2
l = 1                           # Dimension of embedded m-flat submanifold
N = 100                        # Number of samples
subsampler = SphericalRandomWalk(0.5)
subsamples = Int(50)

osampler = OrthogonalNaturalGradient(NegativeLogDensity(), subsampler, subsamples)
insampler = IterativeNaturalGradient(NegativeLogDensity(), subsampler, subsamples)

rng = MersenneTwister(1234)
model = make_model(rng, p)

rng = MersenneTwister(1234)
@time osamples = sample(rng, model, osampler, N, chain_type=MCMCChains.Chains);
Chains(samples)

rng = MersenneTwister(1234)
@time insamples = sample(rng, model, insampler, N, chain_type=MCMCChains.Chains);
Chains(samples)

if(p == 2)
    plot(model, gsamples, 1.0)
end
