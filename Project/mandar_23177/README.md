## Overview
In this project, I have tried to implement the paper "Robust Beamforming for RIS Assisted Wireless Networked Control Systems" by Ma et. al.

## Software Requirements

To run this project, the following software is required:<br/>

MATLAB R2026a<br/>
MATLAB Online (used for development, but local MATLAB should also work)<br/>
CVX (Convex Optimization Toolbox for MATLAB)<br/>

## CVX Installation

This project relies on CVX for solving convex optimization problem.

Download CVX from:<br/>
https://cvxr.com/cvx/<br/>
Extract the folder and navigate to it in MATLAB.<br/>
Run the following command in MATLAB: cvx_setup<br/>
Ensure CVX is properly installed and this command is run before running the code.<br/>

## Dependencies
The implementation uses:<br/>
Basic MATLAB linear algebra functions<br/>
dlqr (Discrete LQR)<br/>
dlyap (Discrete Lyapunov equation)
CVX for optimization<br/>

Make sure your MATLAB installation includes:
Control System Toolbox (for dlqr, dlyap)
