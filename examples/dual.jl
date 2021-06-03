using GenericBayes
import GenericBayes.bregman_generator

struct mygeom<:Bregman end
struct mymodel<:BayesModel end

@vector_param MyParam

bregman_generator(θ::MyParam, geometry::mygeom, m::mymodel) = 0.5 *
    θ.components' * θ.components

# Test the defined function
θ = MyParam([1.4; 7.8; -2.0])
@show bregman_generator(θ, mygeom(), mymodel())

# Test transforming to dual params
η = legendre_dual(θ, mygeom(), mymodel())

@show θ
@show legendre_dual(η, mygeom(), mymodel())
