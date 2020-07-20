using GenericBayes, Test, Distributions

@testset "Multinomial" begin
	# TODO: Start with a fixed seed for RNG
	# TODO: Make it an actual test...
	# Generate some true data
	TrueParam = ProbabilityParameter([0.15; 0.8; 0.05])

	# Setup model parameters
	p = length(TrueParam.p)
	n = 100				# Number of data
	np = 100			# Prior "sample size", i.e. informativeness
	tp = ones(p-1) ./ (p-1)		# Prior mode / mean
	data = rand(Categorical(TrueParam.p), n)
	θ_true = NaturalParameter(TrueParam)
	model = MultinomialConjugate(p, n, np, data, tp)
	@show log_posterior_density(model, θ_true)
end
