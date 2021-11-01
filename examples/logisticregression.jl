using GenericBayes, Distributions, LinearAlgebra, StatsFuns, Plots, MCMCChains, ForwardDiff

p = 10                           # Number of coefficients
n = 100                         # Number of observations (data)

# Generate design matrix
X = rand(Normal(), (n, p))
@show β_true = ones(p)
μ = StatsFuns.logistic.(X * β_true)
y = rand(Product(Bernoulli.(μ)))

# Build model
ρ = 0.99
Σ = diagm(ones(p))
Σ[2,1] = ρ
Σ[1,2] = ρ
model = CanonicalGLM{Bernoulli, Float64}(X, y, zeros(p), Σ)

N = 100                        # Number of samples
subsampler = SphericalRandomWalk(0.5)
subsamples = Int(50)
l = 2                           # Dimension of embedded m-flat submanifold
sampler = ERecursiveOrthogonalGibbs(NegativeLogDensity(), l, subsampler, subsamples)

samples = sample(model, sampler, N, chain_type=MCMCChains.Chains)
chain = Chains(samples)

if(p == 2)
    plot(model, samples, 1.0)
end

pyplot()
autocorplot(chain)

map = max_posterior(model, ones(p))
k = 5
ξ = legendre_dual(map, NegativeLogDensity(), model, k)
@show map
@show w=inverse_legendre_dual(map, NegativeLogDensity(), model, k)
F(x) = bregman_generator(x, NegativeLogDensity(), model)
@show metric(map, F, 5)
@show ForwardDiff.hessian(F, map)
