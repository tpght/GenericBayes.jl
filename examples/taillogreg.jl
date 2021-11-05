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
p = 3
l = 1                           # Dimension of embedded m-flat submanifold
N = 100                        # Number of samples
subsampler = SphericalRandomWalk(0.5)
subsamples = Int(50)

sampler = ERecursiveOrthogonalGibbs(NegativeLogDensity(), l, subsampler, subsamples)
msampler = MRecursiveOrthogonalGibbs(NegativeLogDensity(), l, subsampler, subsamples)
it_sampler = EIterativeOrthogonalGibbs(NegativeLogDensity(), l, subsampler, subsamples)
mit_sampler = MIterativeOrthogonalGibbs(NegativeLogDensity(), l, subsampler, subsamples)
gsampler = OrthogonalGibbs(NegativeLogDensity(), l, subsampler, subsamples)

rng = MersenneTwister(1234)
model = make_model(rng, p)

rng = MersenneTwister(1234)
@time samples = sample(rng, model, sampler, N, chain_type=MCMCChains.Chains);
Chains(samples)

rng = MersenneTwister(1234)
@time it_samples = sample(rng, model, it_sampler, N, chain_type=MCMCChains.Chains);
Chains(it_samples)

rng = MersenneTwister(1234)
@time msamples = sample(rng, model, msampler, N, chain_type=MCMCChains.Chains);
Chains(msamples)

rng = MersenneTwister(1234)
@time mit_samples = sample(rng, model, mit_sampler, N, chain_type=MCMCChains.Chains);
Chains(mit_samples)

rng = MersenneTwister(1234)
@time gsamples = sample(rng, model, gsampler, N, chain_type=MCMCChains.Chains);
Chains(gsamples)

# @show gsamples == it_samples
@show samples == it_samples
@show maximum([norm(msamples[i] - mit_samples[i]) for i in 1:N])
@show msamples == gsamples


@show norm(gsamples[100]- itm_samples[100], 2)

if(p == 2)
    plot(model, gsamples, 1.0)
end

θi = 1.0

η1 = 10.0
η2 = -11.0

ξ = [η1; η2; θi]
θ = inverse_legendre_dual(ξ, NegativeLogDensity(), model, 2)
G = metric(θ, NegativeLogDensity(), model)

function checker(θ1, θ2, η11, η22)
    θ1_full = inverse_legendre_dual([η11; η22; θ1], NegativeLogDensity(), model, 2)
    θ2_full = inverse_legendre_dual([η11; η22; θ2], NegativeLogDensity(), model, 2)
    GenericBayes.logπ(model, θ1_full) - GenericBayes.logπ(model, θ2_full)
end

checker(1.0, 3.0, 1.0, -5.0)

function checker2(η11, θ22, θ33)
    θ1_full = inverse_legendre_dual([η11; θ22; θ33], NegativeLogDensity(), model, 2)
    G = metric(θ1_full, NegativeLogDensity(), model)
    schur = inv(inv(G)[2:3, 2:3])
    logabsdet(schur)[1], G[1,1]
end

checker2(-4.0, 2.0,1.0)

θ_MAP = max_posterior(model, ones(p))
θp = [-10.0; 2.0; 7.0]
-GenericBayes.logπ(model, θp) - divergence(θp, θ_MAP, NegativeLogDensity(), model)

ing = MFlatGibbs(NegativeLogDensity(), [η1])
ing2 = MFlatGibbs(ing, [η2])
inπ = MFlatConditionalGibbs(model, NegativeLogDensity(), [η1], [1.0])
inπ2 = MFlatConditionalGibbs(inπ, ing, [η2], ones(1))
# Metric on this m-flat submanifold is simply G[3,3]...

ming2 = metric([θi], ing2, inπ2)
θ2 = inverse_legendre_dual([η2; θi], ing, inπ, 1)
metric(θ2, ing, inπ)
inv(inv(G)[2:3, 2:3])
logabsdetmetric(θ, NegativeLogDensity(), model, 1, 3)
M = metric(θ, NegativeLogDensity(), model, 1, 3)
log(M)

@show G[3, 3] - ming2[1]

@show dimension(inπ2)

A = GenericBayes.logπ(inπ2, [θi])
B = GenericBayes.logπ(model, θ) - logabsdet(G[1:2,1:2])[1]
C = GenericBayes.logπ(model, θ) - log(G[1,1]) - log(G[2,2])
@show A - B
@show A - C
@show B - C
