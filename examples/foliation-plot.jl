using GenericBayes, Distributions, LinearAlgebra
using StatsFuns, Plots, Random

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

# Generate a 2D logistic regression model
rng = MersenneTwister(1234)
model = make_model(rng, 2)

# First, plot the heatmap of the posterior contours
plt=plot(model, 1.0)

# Draw an e-flat submanifold, i.e. a line
# This has the form θ = A α + b
A = [1.0; 1.0]
θmap = max_posterior(model, zeros(2))
ploteflat!(plt, A, θmap)

# Draw an m-flat submanifold
# Generalized mixed co-ordinates: b is known from above
geometry = NegativeLogDensity()
plotmflat!(plt, A, model, geometry)

display(plt)
