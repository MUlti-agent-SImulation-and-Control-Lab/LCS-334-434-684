## Overview
In this project, I have tried to implement the paper "Robust Beamforming for RIS Assisted Wireless Networked Control Systems" by Ma et. al.

## Software Requirements

To run this project, the following software is required:

MATLAB R2026a 
MATLAB Online (used for development, but local MATLAB should also work)
CVX (Convex Optimization Toolbox for MATLAB)

## CVX Installation

This project relies on CVX for solving convex optimization problem.

Download CVX from:
https://cvxr.com/cvx/
Extract the folder and navigate to it in MATLAB.
Run the following command in MATLAB: cvx_setup
Ensure CVX is properly installed and this command is run before running the code.

## Dependencies
The implementation uses:
Basic MATLAB linear algebra functions
dlqr (Discrete LQR)
dlyap (Discrete Lyapunov equation)
CVX for optimization

Make sure your MATLAB installation includes:
Control System Toolbox (for dlqr, dlyap)
