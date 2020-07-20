export NaturalParameter, MeanParameter

# Natural and mean parameters are related by Laplace transform

@vector_param NaturalParameter
@vector_param MeanParameter
@reparam NaturalParameter MeanParameter (m,x)->∇ψ(m,NaturalParameter(x))
@reparam MeanParameter NaturalParameter (m,x)->∇ϕ(m,MeanParameter(x))

include("Diaconis.jl")
