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
Σ = diagm(ones(p))
Σ[2,1] = ρ
Σ[1,2] = ρ
model = CanonicalGLM{Bernoulli, Float64}(X, y, zeros(p), Σ)

N = 1000                        # Number of samples
subsampler = AdaptiveRejectionSampler()
subsamples = Int(1)
sampler = OrthogonalNaturalGradient(NegativeLogDensity(), subsampler, subsamples)

samples = sample(model, sampler, N, chain_type=MCMCChains.Chains)
chain = Chains(samples)

if(p == 2)
    plot(model, samples, 1.0)
end

pyplot()
autocorplot(chain)

HMC = HamiltonianMonteCarlo(true, 1000)
HMCsamples, stats = sample(model, HMC, N, chain_type=MCMCChains.Chains, progress=true)
HMCchain = Chains(HMCsamples)
autocorplot(HMCchain)

if(p == 2)
    plot(model, HMCsamples, 1.0)
end
