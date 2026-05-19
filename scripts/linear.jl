using GMAM
using Simulator

using StaticArrays, LinearAlgebra, MatrixEquations
using JLD2

const fprefix="linear"
const cwd = pwd()

#############################################
# Define the dynamics
#############################################

# Important that params are const for Enzyme AD
const lam = 0.5
const A = [-lam 1.0; 0 -lam] # Note that A is not symmetric: non-reversible drift
const B = [1 0 ; 0 1]

@debug "The drift matrix: " a=A

# analytic potential to compare with
# A cov + cov A' + 2BB' = 0
const cov = lyapc(A, 2*B*B')  
V(x::AbstractArray) = 1/2 * x ⋅ (cov\x)
∇V(x::AbstractArray) = cov\x

@info "Linear Drift and Noiste" details="""
    Drift matrix A = $(A)
    Noise matrix B = $(B)
    """

const params = (A=A, B=B)

function drift(x::AbstractVector)
    return A*x
end

function sig(x::AbstractVector)
    return B
end

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
    dv = phis[end]
    
    @info "... converged in $iters steps. action=$action"
    @info "Potential and gradient" detail=
    """∇V= $(dv)
    V = $(v)"""

    @info "Solving Riccati equation"
    Rs = Riccati(drift, sig, eq, phis, ps)
    @info "Initial Matrix: " R0 = Rs[1]
    @info "Final Matrix: " R1 = Rs[end]
    @info "Covariance Inverse: " covariance_inv=inv(cov)

    prefac = Prefactor(drift, sig, eq, phis, ps, lambdas, Rs)
    @info "Prefactor" prefac

    return path, V_, ∇V_, prefactor
end

#############################################
# Validation
#############################################

function compute_mc(drift, sig, stable_eq, eps_list)

    # Check if data exists

    # Gen MC 

    return 
end


function main()

    stable_eq = SA{Float32}[0.0, 0.0] # Using StaticArrays for MC sim on GPU
    eval_point = SA{Float32}[ 1.0, 0]

    

end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end