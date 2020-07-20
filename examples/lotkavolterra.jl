using DifferentialEquations, Distributions, GenericBayes, Plots

function lotka_volterra(du,u,p,t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α*x - β*x*y
    du[2] = dy = -δ*y + γ*x*y
end
u0 = [1.0,1.0]
tspan = (0.0,9.9)
p_true = [1.5,1.0,3.0,1.0]
range  = 0.0:0.5:9.9
prior = MvNormal(ones(4), 1.0)

observe(sol) = MvNormal([x[1] for x in sol.u], 0.5)

prob = ODEProblem(lotka_volterra,u0,tspan,p_true)
data = rand(observe(solve(prob, Tsit5(), saveat = range)))

model = ODEModel(prob, Tsit5(), range, observe, data, prior)

θ_true = ODEParameter(p_true)
@show loglikelihood(model, θ_true)
@show ∇logπ(model, θ_true)
@show ∇²logπ(model, θ_true)
