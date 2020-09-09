using LinearAlgebra, GenericBayes, Distributions, Plots

n_p = 100 # Prior sample size
ρ = -0.8  # Prior correlation
Σ_p = [1.0 ρ; ρ 1.0]  # Prior covariance matrix
σ = 1.0  # Standard deviation of observational noise
Σ = Matrix(σ *I, 2, 2)
μ_p = [2.0; 1.0] # Prior mean
n = 100 # Number of data
F = [1.0 0.; 0. 1.]

# Generate some fake data
θ_true = GaussianParam([2.; 2.])
dist = MvNormal(F * θ_true.components, Σ)
y = rand(dist, n)

model = LinearGaussian(n_p, Σ_p, Σ, μ_p, F, y)

xrange=range(1.5, stop=2.1, length=100)
yrange=range(1.0, stop=1.5, length=100)
post_plot=quantile_contour2D(model, typeof(θ_true),xrange,yrange)

# Check that samples line up with contours
samples = sample(model, θ_true, 1000)
scatter!(post_plot, samples)

quantiles = quantile.(Chisq(1), 0.1:0.1:0.9) .- log_posterior_density(model, map(model,θ_true))

# Get value of -lpd at samples
nlpd = -log_posterior_density.(Ref(model), samples)
histogram(nlpd, bins=quantiles)
y = quantile.(Ref(nlpd), 0.1:0.1:0.9)
plot(quantiles, y)



@show ∇logπ(model, map(model, GaussianParam))

