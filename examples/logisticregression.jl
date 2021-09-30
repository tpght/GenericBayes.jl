using GenericBayes, Distributions, LinearAlgebra, StatsFuns, Plots, MCMCChains

p = 2                           # Number of coefficients
n = 100                         # Number of observations (data)

# Generate design matrix
X = rand(Normal(), (n, p))
@show β_true = ones(p)
μ = StatsFuns.logistic.(X * β_true)
y = rand(Product(Bernoulli.(μ)))

# Build model
ρ = 0.99
Σ = [1.0 ρ; ρ 1.0]
model = CanonicalGLM{Bernoulli, Float64}(X, y, zeros(p), Σ)

N = 1000                        # Number of samples
subsampler = SphericalRandomWalk(0.001)
subsamples = Int(5000)
l = 1                           # Dimension of embedded m-flat submanifold
sampler = RecursiveOrthogonalGibbs(NegativeLogDensity(), l, subsampler, subsamples)

samples = sample(model, sampler, N, chain_type=MCMCChains.Chains)
chain = Chains(samples)

if(p == 2)
    plot(model, samples, 1.0)
end

pyplot()
autocorplot(chain)