using GenericBayes, Distributions, LinearAlgebra
using StatsFuns, Plots, ForwardDiff, Random, MCMCChains

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
    CanonicalGLM(X, y, zeros(p), Σ, Bernoulli)
end

# Create samplers
p = 2
block_size = 1
N = 1000                        # Number of samples
# subsampler = ARMS(5.0, true)
subsampler = SphericalRandomWalk(0.75, zeros(1))
subsamples = Int(20)
θ0 = [1.5712046633289665; 1.6388795266403455]
stop_after = 10
#osampler = OrthogonalNaturalGradient(NegativeLogDensity(), subsampler, subsamples)
# sampler = ERecursiveOrthogonalGibbs(NegativeLogDensity(), 1, subsampler, subsamples)
 gsampler = MOrthogonalGradient(NegativeLogDensity(), subsampler, subsamples,
                                θ0,stop_after)
# arms = ARMS(-10.0, 10.0, false)
ggibbssampler = GeneralMGibbs(NegativeLogDensity(), block_size, subsampler,
                              subsamples, θ0)
msampler = MRecursiveOrthogonalGibbs(NegativeLogDensity(), block_size,
                                     subsampler, subsamples, θ0)
hmcsampler = HamiltonianMonteCarlo(true, 100)

rng = MersenneTwister(1234)
model = make_model(rng, p)

rng = MersenneTwister(1234)
gsamples = sample(rng, model, gsampler, N, chain_type=MCMCChains.Chains);
@show min_ess_per_sec(gsamples)
@show min_ess(gsamples)

rng = MersenneTwister(1234)
hmcsamples = sample(rng, model, hmcsampler, N, chain_type=MCMCChains.Chains);
@show min_ess_per_sec(hmcsamples)
@show min_ess(hmcsamples)

# rng = MersenneTwister(1234)
# @time osamples = sample(rng, model, osampler, N, chain_type=MCMCChains.Chains);
# Chains(osamples)

# rng = MersenneTwister(1234)
# @time samples = sample(rng, model, sampler, N, chain_type=MCMCChains.Chains);
# Chains(samples)
θ = 2.0 * ones(p)
geometry = NegativeLogDensity()
A = diagm(ones(2))
Ap = A[:,2:2]
ξ = legendre_dual(θ, geometry, model, 1)
ξ2 = legendre_dual(θ, geometry, model, Ap)
inverse_legendre_dual(ξ, geometry, model, 1)
inverse_legendre_dual(ξ2, [0.0, ξ[2]], Ap, geometry, model)

logabsdetmetric(θ, geometry, model, 1)
logabsdetmetric(θ, geometry, model, Ap)

rng = MersenneTwister(1234)
set_initial(msampler.subsampler, zeros(1))
msamples = sample(rng, model, msampler, N, chain_type=MCMCChains.Chains);

rng = MersenneTwister(1234)
set_initial(ggibbssampler.subsampler, zeros(1))
ggibbssamples = sample(rng, model, ggibbssampler, N, chain_type=MCMCChains.Chains);

# rng = MersenneTwister(1234)
# mitsamples = sample(rng, model, mitsampler, N, chain_type=MCMCChains.Chains);

# rng = MersenneTwister(1234)
# hmcsamples = sample(rng, model, hmcsampler, N, chain_type=MCMCChains.Chains);

function chain_values(chains)
    v = chains.value.data[:,:,1]
    [x for x in eachrow(v)]
end

c1 = chain_values(msamples)
c2 = chain_values(ggibbssamples)
@show c1[2] - c2[2]
maximum(norm.(c1 - c2))

v = chain_values(gsampler)
if(p == 2)
    plot(model, c2, 1.0)
end

# @show maximum(norm(ggibbssamples - samples))
@show maximum(norm(ggibbssamples.value - msamples.value))
# @show maximum(norm(gsamples - osamples))
