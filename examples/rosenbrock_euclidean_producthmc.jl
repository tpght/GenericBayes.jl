using GenericBayes, LinearAlgebra, Distributions, Random, AbstractMCMC,
    StatsBase, StatsPlots, MCMCChains

model = Rosenbrock()

θ1 = RosenbrockParameter(ones(2))
θ1.components[1] = θ1.components[2] + rand()

geometry = Euclidean{typeof(θ1)}()

state = ProductHMCState(θ1, legendre_dual(θ1, geometry, model))

@show hamiltonian(state, geometry, model)
@show grad_hamiltonian(state, geometry, model)

# Test leapfrog
for i in 1:1000
    leapfrog!(state, 0.00001, geometry, model)
end

@show hamiltonian(state, geometry, model)

# Test sampler
method = ProductManifoldHMC{Float64, typeof(θ1)}(geometry, 0.00001, 1000, 1.5)
samples = sample(model, method, 5000, chain_type=MCMCChains.Chains)
chain = Chains(samples)
@show(ess(chain))

samples2 = cat(samples..., dims=2)
x = samples2[1,200:end]
y = samples2[2,200:end]
scatter(x, y) 
