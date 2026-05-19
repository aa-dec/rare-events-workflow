# Workflow for rare events in SDEs

This project contains examples illustrating how to compute rare event 
asymptotics for rare events associated to SDEs 

$$ 
dX_t = b(X_t) \, dt + \sqrt{2 \varepsilon} a(X_t) dW_t. 
$$

Assumptions needed on the system:

 - the drift term $b(x)$ has a stable fixed point $b(x\_*) = 0$ at $x\_*$.
 - the drift term is such that there exists a unique ergodic measure $\rho dx$.
 - the only attractors for the deterministic system $\dot x = b(x)$ are fixed point attractors. 
 - the symmetric matrix $\sigma = a a^\top$ is invertible (full noise model, might be able to relax this... will require some more thought)

## Steps

1. Define the dynamics (need to write it in a way that Enzyme can autodiff) by specifying the drift vector and diffusion matrix
2. Find the stable equilibrium (Newton/Optimization method)
3. Pick a 'rare event' as a state $x$ in the basin of attraction of the stable equilibrium
4. Compute the Instanton path $\phi(s)$, Ricatti matrices $\nabla^2 V(\phi(s))$ endpoint potential $V(x)$ and endpoint derivative $\nabla V(x)$ using GMAM.
5. Use the path data to compute the prefactor $C$ (also in the GMAM library)
6. Small noise asymptotics are given:
$$\log E[T] \approx -\log C + \frac{1}{2} \log \varepsilon + \frac{V}{\varepsilon}.$$

## Things to note: 

Enzyme is rather unstable. Everything was developed in a julia 1.10 environment to enforce using a stable version of Enzyme.

Code Likely requires a GPU to run the MC simulation tests. Comment out the necessary lines in the main functions if you don't have CUDA on your system. 

Run files with
```
julia --project=. scripts/linear.jl 
```

