using GenericBayes, Distributions, LinearAlgebra, StatsFuns, Plots

p = 2                           # Number of coefficients
n = 100                         # Number of observations (data)

# Generate design matrix
X = rand(Normal(), (n, 2))
β_true = [-2.0, 1.5]
μ = StatsFuns.logistic.(X * β_true)
y = rand(Product(Bernoulli.(μ)))

# Build model
model = CanonicalGLM{Bernoulli, Float64}(X, y, zeros(2), diagm(ones(2)))
