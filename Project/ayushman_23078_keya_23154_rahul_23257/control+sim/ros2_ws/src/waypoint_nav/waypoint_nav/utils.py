import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import casadi as ca


def diff_drive_dynamics(t, y, u):
    """
    Kinematic model of diff drive rover
    y = (x, y, theta)
    u = (left wheel speed, right wheel speed)
    """
    theta = y[2]
    vl, vr = u

    v = (vr + vl) / 2.0
    omega = (vr - vl) / ROBOT_WIDTH

    dx = v * np.cos(theta)
    dy = v * np.sin(theta)
    dtheta = omega
    return [dx, dy, dtheta]


# Parameters
Ts = 0.2          # Sampling time (seconds)
N_horizon = 10    # Prediction horizon steps
Nx = 3            # State dimension: [x, y, theta]
Nu = 2            # Control dimension: [vl, vr]

ROBOT_WIDTH = 0.698   # Distance between wheels (meters), Husky A200
WHEEL_RADIUS = 0.1    # Wheel radius (meters)

#
MAX_WHEEL_SPEED = 5.0  
MAX_ANGULAR_VEL = 0.8  


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# DeepONet hyperparams
p = 50
trunk_input_dim = 2              # [cos(theta), sin(theta)]
branch_input_dim = Nu * N_horizon  # 20
output_dim_total = Nx * N_horizon  # 30
epoch = 2500


def get_casadi_mlp(model, inp):
    """Convert a PyTorch MLP to a CasADi symbolic expression."""
    x = inp
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            w = layer.weight.detach().cpu().numpy()
            b = layer.bias.detach().cpu().numpy()
            x = ca.mtimes(w, x) + b
        elif isinstance(layer, nn.Tanh):
            x = ca.tanh(x)
    return x


# MPC cost weights
Q = np.diag([3.0, 3.0, 0.5])   
R = np.diag([0.15, 0.15])     
R_delta = np.diag([10.0, 10.0])  