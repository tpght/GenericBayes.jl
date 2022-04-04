using GenericBayes, Distributions, LinearAlgebra
using Plots, MCMCChains, Random, SparseArrays, StatsPlots

# Set a random seed
rng = MersenneTwister(1234)
# TODO: pass rng to sampler

# Example from Fox (2008)
p = 5

U = spdiagm(Pair(1, rand(p-1)),
            Pair(0, ones(p)))

Λ = U' * U

w = ones(p)
model = GaussianInverse(Λ, w)
sampler = GaussianEliminationSampler(rand(p))
N = 10^6

samples=sample(model, sampler, N)

# Look at samples.value.data to see individual samples
# E.g. 50th sample
plot(samples.value.data[2, :, 1])

# Can also plot samples directly
plot(samples)

# Compute sample covariance
C = cov(samples.value.data[:, :, 1])
@show norm(C - inv(Matrix(Λ)), 2)

# Check if the linear system is satisfied by the mean
m = mean(samples.value.data[:, :, 1], dims=1)'
@show norm(m - (Λ \ w), 2)
