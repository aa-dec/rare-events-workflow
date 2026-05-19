# Workflow for rare events in SDEs

This project contains examples illustrating how to compute rare event 
asymptotics for rare events associated to SDEs 

$$ 
dX_t = b(X_t) \, dt + a(X_t) dW_t. 
$$

Assumptions needed on the system:

 - the drift term $b(x)$ has a stable fixed point $b(x\_*) = 0$ at $x\_*$.
 - the drift term is such that there exists a unique ergodic measure $\rho dx$.
 - the only attractors for the deterministic system $\dot x = b(x)$ are fixed point attractors. 
 - the symmetric matrix $\sigma = a a^\top$ is invertible (full noise model, might be able to relax this... will require some more thought)

## Things to note: 

Enzyme is rather unstable. Everything was developed in a julia 1.10 environment 
