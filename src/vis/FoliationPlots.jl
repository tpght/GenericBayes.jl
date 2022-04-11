export ploteflat!, plotmflat!, plotmflatline!

"""
    ploteflat!(plt, A, θ)

Adds the e-flat geodesic described by A*α + b to plt

# Arguments
- `plt`: the plot object to overwrite
- `A::Vector`: two-dimensional tangent vector to the e-flat submanifold
- `θ::Vector`: a 2d point in the e-flat submanifold
"""
function ploteflat!(plt, A, θ)
    # Convert A to a matrix
    A = reshape(A, (2,1))

    # Project
    b = θ - A * pinv(A) * θ
    αmap = pinv(A) * θ
    h = 0.8
    α1, α2 = αmap .- h, αmap .+ h
    θ1 = A * α1 + b
    θ2 = A * α2 + b
    x = [θ1[1]; θ2[1]]
    y = [θ1[2]; θ2[2]]
    plot!(plt, x,y, legend=false, lw=2, lc=:red)
end

"""
    plotmflatline

Plots the m-geodesic connecting θ1 to θ2.
"""
function plotmflatline!(plt, θ1, θ2, model, geometry; npoints=100, lw=1.5,
                        lc=:blue, quiver=false)
    # Convert to dual co-ordinates.
    η1 = legendre_dual(θ1, geometry, model)
    η2 = legendre_dual(θ2, geometry, model)

    h =  (1.0/ (npoints - 1)) .* (η2 - η1)
    points = [η1 .+ (i-1) .* h for i in 1:npoints]

    # Convert points back to primal co-ordinates.
    for i=1:npoints
        points[i] = inverse_legendre_dual(points[i], geometry, model)
    end

    # Plot the points
    x = [points[i][1] for i in 1:npoints]
    y = [points[i][2] for i in 1:npoints]

    if(quiver==false)
        plot!(plt, x, y, lw=lw, lc=lc)
    else
        plot!(plt, x[1:end-1], y[1:end-1], lw=lw, lc=lc)
        dx = [x[npoints] - x[npoints-1]; y[npoints] - y[npoints-1]]
        quiver!(plt, [x[npoints-1]], [y[npoints-1]],
                quiver=([dx[1]], [dx[2]]), lc=lc)
    end
end

"""
    plotmflat!(plt, A, θ)

Adds the m-flat geodesic described by η = B*β + d to plt

# Arguments
- `plt`: the plot object to overwrite
- `A::Vector`: Euclidean-normal vector to m-flat submanifold
- `θ::Vector`: a 2d point in the m-flat submanifold
- `model::BayesModel`: statistical model
- `geometry::Bregman`: dually-flat geometry
"""
function plotmflat!(plt, A, θ, model, geometry)
    # Convert A to a matrix
    A = reshape(A, (2,1))

    # δ is held constant over the m-flat submanifold
    δ = pinv(A) * legendre_dual(θ, geometry, model)

    # NOTE: Currently inverse_legendre_dual assumes A is orthogonal.
    # Since A here is not orthogonal, we have to transform δ to A^T A δ
    δ = A' * A * δ
    δ = reshape(δ, (1,))

    # xf, yf hold co-ordinates of points on the submanifold
    xf = zeros(0)
    yf = zeros(0)

    Δ = 0.1
    T = 1.0

    # Number of points
    np = Int(round(T / Δ))

    for i=1:np
        bp = (i * Δ) .* b
        θ11 = inverse_legendre_dual(δ, bp, A, geometry, model, x0=[α])
        θ22 = inverse_legendre_dual(δ, -bp, A, geometry, model, x0=[α])
        push!(xf, θ11[1])
        push!(yf, θ11[2])
        pushfirst!(xf, θ22[1])
        pushfirst!(yf, θ22[2])
    end

    plot!(plt, xf, yf, legend=false, lw=2, lc=:blue)
end
