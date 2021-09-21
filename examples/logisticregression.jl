using GenericBayes, Distributions, LinearAlgebra, StatsFuns, Plots, MCMCChains

p = 2                           # Number of coefficients
n = 1000                         # Number of observations (data)

# Generate design matrix
X = rand(Normal(), (n, 2))
@show β_true = [-2.0, 1.5]
μ = StatsFuns.logistic.(X * β_true)
y = rand(Product(Bernoulli.(μ)))

# Build model
model = CanonicalGLM{Bernoulli, Float64}(X, y, zeros(2), diagm(ones(2)))

@show loglikelihood(model, β_true)
@show log_posterior_density(model, β_true)
@show map = max_posterior(model, zeros(2))
@show mle(model, zeros(2))

pyplot()
plot(model, 1.0)

N = 1000                        # Number of samples
step_size = 0.3
samples = sample(model, SphericalRandomWalk(step_size), N, chain_type=MCMCChains.Chains)
chain = Chains(samples)

plot(model, samples, 0.5)
