module GenericBayes

using StatsBase, LinearAlgebra, ForwardDiff, Optim, AbstractMCMC, Distributions
using StatsFuns, MCMCChains, Plots
import Base.Vector, Base.Array, Base.length
import AbstractMCMC.AbstractModel
import Distributions.sample, Distributions.loglikelihood
import Plots.scatter!

export Multinomial, BayesModel, Chart
export log_posterior_density, sufficient_statistic, logπ
export ∇logπ, grad_log_posterior_density
export ∇²logπ, hessian_log_posterior_density
export likelihood, loglikelihood, mle, max_posterior, prior_mode, prior, simulate
export log_posterior_density, dimension

"""
    BayesModel

A `BayesModel` is any type representing a Bayesian model.

Any `BayesModel` should implement at least `log_posterior_density`.
"""
abstract type BayesModel <: AbstractModel end


# Include files from project
include("opt/Opt.jl")
include("models/Models.jl")
include("vis/Density.jl")
include("geometry/Geometry.jl")
include("samplers/Samplers.jl")

"""
A Julia package for writing MCMC samplers independently of model
parameterization or representation.
"""
end # module
