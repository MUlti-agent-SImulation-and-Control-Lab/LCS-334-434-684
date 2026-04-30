## Overview

This project studies a distributed continuous-time optimization algorithm for multi-agent systems operating over strongly connected, weight-balanced directed graphs. The goal is to minimize a global objective formed as the sum of local cost functions using only local interactions.

The work focuses on designing an adaptive algorithm that combines gradient-based optimization with consensus dynamics, while introducing adaptive coupling gains to handle limited network information.

We analyze the system by deriving equilibrium conditions and proving convergence using Lyapunov-based methods, showing that all agents reach consensus at the global optimum.

Simulations (implemented in MATLAB) are used to validate the theoretical results on representative directed graph topologies.

1. requirements.txt contains the requirments for running the code (just requires MATLAB)
2. src/ - contains code for both the simulations, simply donwload the file and run it on MATLAB.
3. results/ - contain the plots generated out of simulation 1 and simulation 2.
## Results - (consensus point)
   ### Simulation Results

| Simulation | Consensus point |
|------------|-------------------------|
| 1          | (-0.2399, 1.0714)       |
| 2          | (-0.8832, 0.7879)       |
