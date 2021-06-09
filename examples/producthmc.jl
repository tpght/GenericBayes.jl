using GenericBayes, LinearAlgebra, Distributions

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

geometry = NegativeLogDensity{GaussianParam{Float64}}()

θ1 = GaussianParam(ones(2))
θ2 = GaussianParam(2.0 * ones(2))

state = ProductHMCState(θ1, legendre_dual(θ2, geometry, model))

@show hamiltonian(state, geometry, model)
@show grad_hamiltonian(state, geometry, model)
