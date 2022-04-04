export ∇ψ, ∇ϕ

function sufficient_statistics(model::DiaconisConjugate{Normal},
                               y::Vector)
    [-mean(y.^2); mean(y)]
end

"""
    ψ(model, θ)

Cumulant generating function in natural parameters
"""
function ψ(model::DiaconisConjugate{Normal},
           θ::Vector) where D<:Distribution
    ((θ[2]^2) / (4.0 * θ[1])) + 0.5 * log(pi / θ[1])
end

"""
    ∇ψ(model, θ)

Gradient of cumulant generating function in natural parameters
"""
function ∇ψ(model::DiaconisConjugate{Normal},
            θ::Vector)
    [-0.25 * (θ[2] / θ[1])^2 - 0.5 * θ[1]^(-1);
     0.5 * (θ[2] / θ[1])]
end

"""
    ∇²ψ(model, θ)

Hessian of cumulant generating function in natural parameters
"""
function ∇²ψ(model::DiaconisConjugate{Normal},
             θ::Vector)
    od = -0.5 * θ[2] / (θ[1]^2) # off-diagonal
    [0.5 * ((θ[2]^2) / θ[1]^3) + 0.5 * θ[1]^(-2) od;
     od 0.5 * (1.0 / θ[1])]
end

"""
    ∇ϕ(model, η)

Gradient of Legendre dual of cumulant generating function
"""
function ∇ϕ(model::DiaconisConjugate{Normal},
            η::Vector)
    [-0.5 / (η[1] + η[2]^2); -η[2] / (η[1] + η[2]^2)]
end

"""
    ∇ϕ(model, ξ, k)

Returns θ where first k-mixed co-ordinates ξ are known
"""
function ∇ϕ(model::DiaconisConjugate{Normal},
            ξ::Vector, k::Int)
    if(k == 2) return ∇ψ(model, ξ) end
    if(k == 0) return ξ end

    # if k == 1, we know θ2=ξ[2] and η1=ξ[1].
    θ = copy(ξ)
    θ[1] = -0.25 * (1.0 / ξ[1]) * (1.0 + sqrt(1.0 - 4.0 * ξ[1] * ξ[2]^2))
    θ
end
