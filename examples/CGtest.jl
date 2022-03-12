using GenericBayes, Distributions, LinearAlgebra
using Plots, MCMCChains, Random, SparseArrays

# Set a random seed
rng = MersenneTwister(1234)
# TODO: pass rng to sampler


# Example from Fox (2008)
p = 5

U = spdiagm(Pair(1, rand(p-1)),
            Pair(0, ones(p)))

Λ = U' * U


w = zeros(p)
model = GaussianInverse(Λ, w)
sampler = ConjugateGradientSampler(rand(p))
N = 10^4

samples=sample(model, sampler, N)

# Look at samples.value.data to see individual samples
# E.g. 50th sample
plot(samples.value.data[2, :, 1])

# Compute sample covariance
C = cov(samples.value.data[:, :, 1])
@show norm(C - inv(Matrix(Λ)), 2)
