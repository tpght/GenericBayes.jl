using GenericBayes, LinearAlgebra, Distributions, Random, AbstractMCMC,
    StatsBase, StatsPlots, MCMCChains

model = Rosenbrock()

# Test sampler
method = SphericalRandomWalk{Float64, RosenbrockParameter{Float64}}(0.05)
samples = sample(model, method, 1000, chain_type=MCMCChains.Chains)
chain = Chains(samples)
@show(ess(chain))

samples2 = cat(samples..., dims=2)
x = samples2[1,200:end]
y = samples2[2,200:end]
scatter(x, y) 
