module GenericBayes

using StatsBase, LinearAlgebra, ForwardDiff, Optim, AbstractMCMC, Distributions
using StatsFuns, MCMCChains, Plots, LineSearches, AdvancedHMC, Random, JTwalk
import Base.Vector, Base.Array, Base.length
import AbstractMCMC.AbstractModel, AbstractMCMC.bundle_samples
import Distributions.sample, Distributions.loglikelihood
import Plots.scatter!

export Multinomial, BayesModel, Chart
export log_posterior_density, sufficient_statistic, logπ
export ∇logπ, grad_log_posterior_density
export ∇²logπ, hessian_log_posterior_density
export likelihood, loglikelihood, mle, max_posterior, prior_mode, prior, simulate
export log_posterior_density, dimension, min_ess, min_ess_per_sec

"""
    BayesModel

A `BayesModel` is any type representing a Bayesian model.

Any `BayesModel` should implement at least `log_posterior_density`.
"""
abstract type BayesModel <: AbstractModel end
abstract type SubmanifoldConditional <: BayesModel end
abstract type MFlatConditional <: SubmanifoldConditional end
abstract type EFlatConditional <: SubmanifoldConditional end
function ambient_model(model::SubmanifoldConditional) end
function dimension(model::BayesModel) end

# Include files from project
include("opt/Opt.jl")
include("geometry/Geometry.jl")
include("models/Models.jl")
include("vis/Density.jl")
include("samplers/Samplers.jl")

function bundle_samples(
    samples,
    model::BayesModel,
    sampler::AbstractMCMC.AbstractSampler,
    current_state::Any,
    ::Type;
    kwargs...
)
    stats = kwargs[:stats]
    chain = Chains(samples)
    setinfo(chain, (start_time=stats.start, stop_time=stats.stop))
end

function min_ess(chain::MCMCChains.Chains)
    minimum(ess_rhat(chain).nt[:ess])
end

function min_ess_per_sec(chain::MCMCChains.Chains)
    minimum(ess_rhat(chain).nt[:ess_per_sec])
end

"""
A Julia package for writing MCMC samplers independently of model
parameterization or representation.
"""
end # module
