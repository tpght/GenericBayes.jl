using Plots

export plot_posterior_density2D

# TODO: Can I just pass optional arguments into contour?
""" Plots the posterior density of a model. """
function plot_posterior_density2D(model::BayesModel, ParameterType::Type{<:Parameter}, xrange, yrange)
    gr()
    post(x1, x2) = log_posterior_density(model, ParameterType([x1, x2]))

    posterior=contour(xrange,yrange,post, title="Log Posterior Density")
    posterior
end
