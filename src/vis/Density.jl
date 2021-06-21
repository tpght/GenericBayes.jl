using Plots

export quantile_contour2D

""" Plots the posterior density of a model. """
function quantile_contour2D(model::BayesModel, ParameterType::Type{<:Parameter}, xrange, yrange, p = 0.1:0.1:0.9)
    # Start plotting backend
    pyplot()
    
    post(x1, x2) = -log_posterior_density(model, ParameterType([x1, x2]))

    # Compute contours using Laplace approximation (see thesis)
    nlpd_at_map = -log_posterior_density(model, max_posterior(model, ParameterType(ones(2))))
    levels =  nlpd_at_map .+ quantile.(Chisq(dimension(model) - 1), p)

    contour(xrange,yrange,post, levels=levels, title="Log Posterior Density")
end

function scatter!(plot_, samples::Vector{P}) where P<:Parameter
    data = Array(samples)
    x = data[1,:]
    y = data[2,:]
    scatter!(plot_, x, y) 
end
