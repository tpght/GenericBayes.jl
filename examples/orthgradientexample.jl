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
block_size = 1
N = 5000                        # Number of samples
subsampler = ARMS(2.0, false)
subsamples = Int(1)
θ0 = zeros(p)
# osampler = OrthogonalNaturalGradient(NegativeLogDensity(), subsampler, subsamples)
# sampler = ERecursiveOrthogonalGibbs(NegativeLogDensity(), 1, subsampler, subsamples)
gsampler = GeneralNaturalGradient(NegativeLogDensity(), subsampler, subsamples, θ0)
# arms = ARMS(-10.0, 10.0, false)
# ggibbssampler = GeneralMGibbs(NegativeLogDensity(), block_size, subsampler, subsamples)
# mitsampler = MRecursiveOrthogonalGibbs(NegativeLogDensity(), block_size, subsampler, subsamples)
# hmcsampler = HamiltonianMonteCarlo(true, 100)

rng = MersenneTwister(1234)
model = make_model(rng, p)

rng = MersenneTwister(1234)
gsamples = sample(rng, model, gsampler, N, chain_type=MCMCChains.Chains);

# rng = MersenneTwister(1234)
# @time osamples = sample(rng, model, osampler, N, chain_type=MCMCChains.Chains);
# Chains(osamples)

# rng = MersenneTwister(1234)
# @time samples = sample(rng, model, sampler, N, chain_type=MCMCChains.Chains);
# Chains(samples)

# rng = MersenneTwister(1234)
# @time msamples = sample(rng, model, msampler, N, chain_type=MCMCChains.Chains);
# Chains(msamples)

# rng = MersenneTwister(1234)
# ggibbssamples = sample(rng, model, ggibbssampler, N, chain_type=MCMCChains.Chains);
# Chains(ggibbssamples)

# rng = MersenneTwister(1234)
# mitsamples = sample(rng, model, mitsampler, N, chain_type=MCMCChains.Chains);

# rng = MersenneTwister(1234)
# hmcsamples = sample(rng, model, hmcsampler, N, chain_type=MCMCChains.Chains);


v = gsamples.value.data[:,:,1]
v = [x for x in eachrow(v)]

if(p == 2)
    plot(model, v, 1.0)
end

# @show maximum(norm(ggibbssamples - samples))
@show maximum(norm(ggibbssamples.value - mitsamples.value))
# @show maximum(norm(gsamples - osamples))
