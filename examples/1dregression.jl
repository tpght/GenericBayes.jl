using GenericBayes, Distributions, LinearAlgebra, StatsFuns, Plots, MCMCChains

p = 1                           # Number of coefficients
n = 100                         # Number of observations (data)

# Generate design matrix
X = rand(Normal(), (n, p))
@show β_true = ones(p)
μ = StatsFuns.logistic.(X * β_true)
y = rand(Product(Bernoulli.(μ)))

model = CanonicalGLM{Bernoulli, Float64}(X, y, zeros(p), diagm(ones(p)))


ℓπ(θ) = log_posterior_density(model, [θ])
∇ℓπ(θ) = ∇logπ(model, [θ])[1]

map = max_posterior(model, ones(1))

N = 1000                        # Number of samples
sampler = AdaptiveRejectionSampler()
samples = sample(model, sampler, N, chain_type=MCMCChains.Chains)
chain = Chains(samples)

pyplot()
autocorplot(chain)
histogram(chain)
