using GenericBayes, LinearAlgebra, Distributions, Random, AbstractMCMC,
    StatsBase, StatsPlots, MCMCChains

n_p = 10 # Prior sample size

ρ = -0.8  # Prior correlation
Σ_p = [1.0 ρ; ρ 1.0]  # Prior covariance matrix
σ = 1.0  # Standard deviation of observational noise
Σ = Matrix(σ *I, 2, 2)
μ_p = [2.0; 1.0] # Prior mean
n = 10 # Number of data
F = [1.0 0.; 0. 1.]

# Generate some fake data
θ_true = GaussianParam([17.0; -2.00])
dist = MvNormal(F * θ_true.components, Σ)
y = rand(dist, n)
model = LinearGaussian(n_p, Σ_p, Σ, μ_p, F, y)

geometry = Euclidean{GaussianParam{Float64}}()

θ1 = GaussianParam(ones(2))
θ2 = GaussianParam(2.0 * ones(2))

state = ProductHMCState(θ1, legendre_dual(θ2, geometry, model))

@show hamiltonian(state, geometry, model)
@show grad_hamiltonian(state, geometry, model)

# Test leapfrog
method = ProductManifoldHMC{Float64, GaussianParam{Float64}}(geometry, 0.01, 12, 1.5)
samples = sample(model, method, 5000, chain_type=MCMCChains.Chains)
chains = Chains(samples)
@show(ess(chains))

xrange=range(8.0, stop=10.0, length=100)
yrange=range(-5.0, stop=-3.0, length=100)
post_plot = quantile_contour2D(model, typeof(θ_true), xrange, yrange)

samples2 = cat(samples..., dims=2)

x = samples2[1,200:end]
y = samples2[2,200:end]
scatter!(post_plot, x, y) 
@show ess(chains)
