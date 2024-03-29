using GenericBayes, Distributions, LinearAlgebra
using StatsFuns, Plots, MCMCChains, ForwardDiff, Random

# Set a random seed
rng = MersenneTwister(1234)

p = 2                           # Number of coefficients
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

N = 1000                        # Number of samples
subsampler = SphericalRandomWalk(0.5)
subsamples = Int(100)
l = 2                           # Dimension of embedded m-flat submanifold
sampler = ERecursiveOrthogonalGibbs(NegativeLogDensity(), l, subsampler, subsamples)

@time samples = sample(model, sampler, N, chain_type=MCMCChains.Chains)
chain = Chains(samples)

ess=ess_rhat(chain)

HMCsamples = sample(model, HamiltonianMonteCarlo(true, 100), N,
                    chain_type=MCMCChains.Chains)
HMCchain = Chains(HMCsamples)

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
