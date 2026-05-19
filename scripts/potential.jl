using GMAM
using Simulator
using Plots

using StaticArrays, LinearAlgebra, MatrixEquations, Optim
using JLD2

const fprefix="potential"
const cwd = pwd()

gr()
#############################################
# Define the dynamics
#############################################


@info "Testing against an analytic example: Nonlinear potential drift."
const B = [1 0 ; 0 1]
const ϵ = 1f-1
const params = (B=SMatrix{2, 2, Float32}(B),)

"""
    V(u::Vector{Float64})
Potential with wells at (-1, -1) and (1, 0), biased toward (1, 0).
"""
function V(u::Vector{Float64})
    x, y = u[1], u[2]
    return ((x-1)^2 + y^2) * ((x+1)^2 + (y+1)^2) + ϵ * (x-1)^2
end

V(x::Float64, y::Float64) = V([x,y])
"""
    ∇V(u::Vector{Float64})
Gradient vector: [dV/dx, dV/dy]
"""
function ∇V(u::AbstractVector)
    x, y = u[1], u[2]
    dx = 2*ϵ*(x - 1) + 4*x^3 + 4*x*y^2 + 4*x*y - 2*x - 4*y - 2
    dy = 2*y*((x + 1)^2 + (y + 1)^2) + 2*(y + 1)*(y^2 + (x - 1)^2)
    return [dx, dy]
end

"""
    ∇²V(u::AbstractVector)
2x2 Hessian matrix
"""
function ∇²V(u::AbstractVector)
    x, y = u[1], u[2]
    
    h11 = 2*ϵ + 12*x^2 + 4*y^2 + 4*y - 2
    h12 = 8*x*y + 4*x - 4
    h22 = 4*x^2 + 12*y^2 + 12*y + 6
    
    return [h11 h12; 
            h12 h22]
end

# --- In-place versions for performance ---

function ∇V!(dv::AbstractVector, u::AbstractVector)
    x, y = u[1], u[2]
    dv[1] = 2*ϵ*(x - 1) + 4*x^3 + 4*x*y^2 + 4*x*y - 2*x - 4*y - 2
    dv[2] = 2*y*((x + 1)^2 + (y + 1)^2) + 2*(y + 1)*(y^2 + (x - 1)^2)
end

function ∇²V!(ddv::Matrix{Float64}, u::AbstractVector)
    x, y = u[1], u[2]
    ddv[1,1] = 2*ϵ + 12*x^2 + 4*y^2 + 4*y - 2
    ddv[1,2] = 8*x*y + 4*x - 4
    ddv[2,1] = ddv[1,2]
    ddv[2,2] = 4*x^2 + 12*y^2 + 12*y + 6
end

function drift(x::AbstractVector) 
    return - ∇V(x)
end
function sig(x::AbstractVector) 
    return B 
end

function drift_mc(u::AbstractVector, p, t) # Need to hardcode things for the GPU
    x, y = u[1], u[2]
    dx = 2*ϵ*(x - 1) + 4*x^3 + 4*x*y^2 + 4*x*y - 2*x - 4*y - 2
    dy = 2*y*((x + 1)^2 + (y + 1)^2) + 2*(y + 1)*(y^2 + (x - 1)^2)
    return -SA[dx, dy]
end
function sig_mc(u, p, t) # Need to hardcode things for the GPU
    s2 = sqrt(2.0f0)
    du1 = s2 * p.B[1,1]
    du2 = s2 * p.B[2,2]
    return SA[du1, du2]
end

# Finding stable equilibrium near (1, 0)
function find_minimum(start_u)
    func = Optim.TwiceDifferentiable(V, ∇V!, ∇²V!, start_u)
    res = Optim.optimize(func, start_u, NewtonTrustRegion())

    return res
end

initial_u = [1.0, 0]
results = find_minimum(initial_u)

if Optim.converged(results)
    @info "Minimum found at: ", Optim.minimizer(results)
    @info "Function value: ", Optim.minimum(results)
else
    @info "Optimization failed to converge."
end

eq = Optim.minimizer(results)
endpt = [0.0, -0.0]

#############################################
# GMAM
#############################################

function run_gmam(drift, sig, stable_eq, endpt)

    println("-"^60)
    println("Computing Instanton between:")
    println("  Stable Equilibrium (x*): ", stable_eq)
    println("  Target Endpoint    (xT): ", endpt)
    println()
    println("-"^60)

    v, phis, ps, lambdas, action_list, iter = Instanton(drift, sig, stable_eq, endpt)
    dv = ps[end]
    
    @info "... converged in $iter steps. action=$v"
    @info "Potential and gradient" detail=
    """∇V= $(dv)
    V = $(v)"""

    @info "Solving Riccati equation"
    Rs = Riccati(drift, sig, stable_eq, phis, ps)
    @info "Initial Matrix: " R0 = Rs[1]
    @info "∇²V(stable_eq): " ∇²V(stable_eq)
    @info "Final Matrix: " R1 = Rs[end]
    @info "∇²V(endpoint): " ∇²V(endpt)

    prefac = Prefactor(drift, sig, stable_eq, phis, ps, lambdas, Rs)
 
    @info "Prefactor" prefac
    println("-"^60)
    return phis, v, dv, prefac
end

#############################################
# Validation
#############################################

function run_mc(drift_mc, sig_mc, startpt, endpt, eps_list, force_sim=false)
    dim = length(startpt)
    startpt = SVector{dim, Float32}(startpt)
    endpt   = SVector{dim, Float32}(endpt)
    n_traj = 1000
    T_end = 1000.0f0
    dt = 0.01f0
    dim = 2
    n_eps = length(eps_list)

    println("-"^60)
    println("Running a Monte Carlo simulation to check hitting time for plane:")
    println("   normal:  n = ∇V = ", ∇V(endpt))
    println("   Target Endpoint:    (xT)= ", endpt)
    println("   Extreme domain:     ∇V(endpt) ⋅ (x - endpt) ≥ 0")
    println()
    println("   Number of trajectories: ", n_traj)
    println("   End Time: ", T_end)
    println("   dt: ", dt)
    println("   Number of eps: ", n_eps)
    println("-"^60)

    # Check if data exists
    save_dir = joinpath(cwd, "data")
    fname = fprefix * "-MC.dat"
    pathname = joinpath(save_dir, fname)
    if isfile(pathname) && !force_sim
        @info "Found previous sim data... loading that. To force a new simulation, set force_sim=true"
        data_mc = load(joinpath(cwd, "data", fname))
        if !(data_mc["endpt"] == endpt) 
            @warn "Loaded endpoint differst from given, expect different asymptotics."
        end
        return data_mc["log_return_stat"], data_mc["log_p_hit"], data_mc["log_conditional"], data_mc["log_first_hit"], data_mc["eps_list"], data_mc["endpt"]
    end

    # Gen MC 

    log_return_stat = Array{Float64}(undef, n_eps)
    log_first_hit = Array{Float64}(undef, n_eps)
    log_p_hit = Array{Float64}(undef, n_eps)
    log_conditional = Array{Float64}(undef, n_eps)

    for (i, ep)  in enumerate(eps_list)
        @info "Epsilon: " ep=ep
        p = (B = sqrt(ep) * params.B, )
        potential_mc = SDEExperiment(
            drift_mc,
            sig_mc,
            p; 
            n_traj=n_traj, T_end=T_end, dt=dt, dim=dim
        )
        inits = [startpt for _ in 1:n_traj]
        trajs = run_experiment(potential_mc, inits) # (dim, n_timesteps, n_trajs)

        # compute expected return time from MC sim
        # Using the Kac recurrence relation E[T|B] = dt/\mu(B)

        @info "Lestang's First hitting statistic" #https://arxiv.org/abs/1711.08428
        function extreme_indicator(x, eval_point)
            return ∇V(eval_point) ⋅ (x - eval_point) ≥ 0
        end

        count_no_hits = 0
        return_stat_ave = 0
        first_hit_ave = 0
        for i in 1:n_traj 
            traj = trajs[i]
            hits = findall(state_vec -> extreme_indicator(state_vec, endpt), eachcol(traj)) 
            if isempty(hits)
                count_no_hits +=1
                continue
            end

            push!(hits, 0)
            sort!(hits)
            time_diffs = dt*(hits[2:end] - hits[1:end-1])
            first_hit_ave += time_diffs[1]
            
            return_stat = 1/(2*T_end) * (sum(time_diffs.^2))
            return_stat_ave += return_stat

        end

        first_hit_ave = first_hit_ave/(n_traj - count_no_hits)
        return_stat_ave = return_stat_ave/(n_traj - count_no_hits) #E[T | hit]
        p_hit = (n_traj - count_no_hits)/n_traj
        if count_no_hits == n_traj
            conditional = Inf
            log_conditional[i] = Inf
            first_hit_ave = Inf
        else
            # E[T] = E[T |hit] P(hit) + E[T| no hit]P(no hit)
            # E[T| no hit] ≈ (T_end + E[T])     Markov property (sort of)
            # => E[T] ≈ E[T | hit] + T_end (P(no hit)/P(hit))
            conditional = return_stat_ave + T_end* (1-p_hit)/p_hit  
            log_conditional[i] = log(conditional)
            first_hit_ave = first_hit_ave + T_end*(1-p_hit)/p_hit
        end
        log_return_stat[i] = log(return_stat_ave)
        log_p_hit[i] = log(p_hit)
        log_first_hit[i] = log(first_hit_ave)

        @info "Hitting prob: " p=p_hit
        @info "statistic: " stat=return_stat_ave
        @info "conditional debiasing: " debias=conditional
        @info "first_hit_ave: " first_hit=first_hit_ave
    end

    
    @info "Saving file in data\\($fname)"
    mkpath(save_dir)
    jldsave(pathname; log_return_stat, log_p_hit, log_conditional, log_first_hit, eps_list, endpt, dt)

    println("-"^60)

    return log_return_stat, log_p_hit, log_conditional, log_first_hit, eps_list, endpt
end


function main()

    stable_eq = eq 
    eval_point = [ 0.5, 0.0]
    log10_min_eps = -1.2
    n_eps = 10
    eps_list = Float32.(10.0 .^range(0, log10_min_eps, n_eps))

    log_return_stat, log_p_hit, log_conditional, log_first_hit, eps_list, endpt = run_mc(drift_mc, sig_mc, stable_eq, eval_point, eps_list,true)
    path, V_, ∇V_, prefactor = run_gmam(drift, sig, stable_eq, eval_point)
    
    # Plotting analytic V, level sets, and extreme region
    L = 1.5
    n_grid = 30

    x_grid = -L:(1/n_grid):L
    y_grid = -L:(1/n_grid):L

    line(x::AbstractArray) = ∇V_'*x 
    line(x::Float64,y::Float64) = line([x,y])

    n_levels = 5
    levels = [V((1-t)*stable_eq + t*(eval_point)) for t in 0:1/n_levels:1]

    heatmap(x_grid, y_grid, V, c=:magma, colorbar = false)
    contour!(x_grid, y_grid, V, levels=levels, color=:white, lw=1) 
    contour!(x_grid, y_grid, line, levels=[∇V_'eval_point], color=:green, lw=1)
    plot!(first.(path), last.(path), color=:blue, lw=2, label="Instanton")

    plot!(title = "Instanton over analytic potential", xlabel = "x", ylabel = "y")
    plot!(xlims = (-L, L), ylims = (-L, L))
    savefig(joinpath(cwd, "plots", fprefix*"-instanton.svg"))


    log_eps = log.(eps_list)
    log_rates = (-1/2*log_eps .+ log(prefactor)) .- (V_ ./ exp.(log_eps)) 
    log_rates_no_prefactor = - (V_ ./ exp.(log_eps)) 
    log_return = - log_rates 

    plot(log_eps, log_return, color=:red, lw=1, label="my formula")
    plot!(log_eps, -log_rates_no_prefactor, color=:black, lw=1, label="no prefactor")
    plot!(log_eps, log_return_stat, color=:green, lw=1, label="Lestang Stat")
    plot!(log_eps, log_conditional, color=:blue, lw=1, label="debiased Lestang")
    plot!(log_eps, log_first_hit, color=:purple, lw=1, label="debiased ave_first_hit")
    plot!(xlabel="log ε", ylabel="log T", title="Return Time Estimates")
    savefig(joinpath(cwd, "plots", fprefix*"-return.svg"))

end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end