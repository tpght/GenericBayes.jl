import Plots.plot

""" Plots the posterior density of a model. """
function plot(model::BayesModel, fill=false, scale=1.0, p = 0.0:0.09:0.99, npoints = 100)
    post(x1, x2) = log_posterior_density(model, [x1; x2])

    # Compute contours using Laplace approximation (see thesis)
    map = max_posterior(model, ones(2))
    lpd_at_map = log_posterior_density(model, map)
    levels =  lpd_at_map .- 0.5 .* quantile.(Chisq(dimension(model)), p)
    sort!(levels)

    xrange = LinRange(map[1]-scale, map[1]+scale, npoints)
    yrange = LinRange(map[2]-scale, map[2]+scale, npoints)
    contour(xrange,yrange,post, levels=levels, fill=fill, title="Log Posterior
        Density", legend=false)
end


function plot(model::BayesModel, samples::Vector, scale=1.0, p =
    0.0:0.09:0.99, npoints = 100)
    plot(model, scale, p, npoints)
    scatter!([sample[1] for sample in samples], [sample[2] for sample in samples])
end
