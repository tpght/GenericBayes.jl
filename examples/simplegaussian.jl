using GenericBayes, Distributions, LinearAlgebra, StatsFuns, Plots, MCMCChains

# Build model
ρ = 0.99
model = SimpleGaussian(ρ)

N = 1000                        # Number of samples
subsampler = SphericalRandomWalk(2.38 * sqrt(1.0 - ρ^2))
subsamples = Int(5000)
l = 1                           # Dimension of embedded m-flat submanifold
sampler = ERecursiveOrthogonalGibbs(NegativeLogDensity(), l, subsampler, subsamples)

samples = sample(model, sampler, N, chain_type=MCMCChains.Chains)
chain = Chains(samples)

pyplot()
plot(model, samples, 1.0)

autocorplot(chain)
