using LinearAlgebra, GenericBayes, Distributions

n_p = 100 # Prior sample size
ρ = -0.8  # Prior correlation
Σ_p = [1.0 ρ; ρ 1.0]  # Prior covariance matrix
σ = 1.0  # Standard deviation of observational noise
Σ = Matrix(σ *I, 2, 2)
μ_p = [2.0; 1.0] # Prior mean
n = 100 # Number of data
F = [1.0 0.; 0. 1.]

# Generate some fake data
θ_true = GaussianParam([17.0; -2.00])
dist = MvNormal(F * θ_true.components, Σ)
y = rand(dist, n)
model = LinearGaussian(n_p, Σ_p, Σ, μ_p, F, y)

geometry = NegativeLogDensity{GaussianParam{Float64}}()

@show bregman_generator(θ_true, geometry, model)
η_true = legendre_dual(θ_true, geometry, model)
@show θ_true
@show legendre_dual(η_true, geometry, model)
