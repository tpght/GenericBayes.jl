using GenericBayes, Distributions, LinearAlgebra, StatsFuns, Plots, MCMCChains

p = 10                           # Number of coefficients
n = 100                         # Number of observations (data)

# Generate design matrix
X = rand(Normal(), (n, p))
@show β_true = ones(p)
μ = StatsFuns.logistic.(X * β_true)
y = rand(Product(Bernoulli.(μ)))

# Build model
model = CanonicalGLM{Bernoulli, Float64}(X, y, zeros(p), diagm(ones(p)))

N = 1000                        # Number of samples
subsampler = SphericalRandomWalk(0.1)
subsamples = Int(100)
l = 5                           # Dimension of embedded m-flat submanifold
sampler = RecursiveOrthogonalGibbs(Euclidean(), l, subsampler, subsamples)

samples = sample(model, sampler, N, chain_type=MCMCChains.Chains)
chain = Chains(samples)

if(p == 2)
    plot(model, samples, 0.5)
end

pyplot()
autocorplot(chain)
