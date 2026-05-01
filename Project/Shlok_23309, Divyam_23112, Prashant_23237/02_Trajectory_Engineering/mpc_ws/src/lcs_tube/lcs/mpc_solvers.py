"""
MPC Solver Classes for ROS2 Controller Nodes

Three controller variants:
1. StandardMPC - Solves at every timestep, no tube, tight coupling to true state
2. TubeMPC - Solves at every timestep, with tube constraints and auxiliary controller
3. ETTubeMPC - Event-triggered: only solves when error exits invariant set

All solvers use CVXPY with OSQP for real-time performance.
"""

import numpy as np
import cvxpy as cp
import scipy.linalg as la
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class StandardMPC:
    """Standard MPC solver - baseline without tube constraints."""
    
    def __init__(self, horizon: int = 20, dt: float = 0.05,
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None,
                 tube_data_path: Optional[str] = None):
        """
        Initialize Standard MPC solver.
        
        Args:
            horizon: MPC prediction horizon N
            dt: Sample time (seconds)
            Q: State cost matrix (6x6)
            R: Input cost matrix (3x3)
            tube_data_path: Path to tube_data.npz (for system matrices only)
        """
        self.N = horizon
        self.dt = dt
        self.nx = 6
        self.nu = 3
        
        # Default weights
        self.Q = Q if Q is not None else np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
        self.R = R if R is not None else np.diag([0.1, 0.1, 0.1])
        
        # Load system matrices and constraints
        if tube_data_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            tube_data_path = str(base_dir / 'simulations' / 'tube_data.npz')
        
        data = np.load(tube_data_path, allow_pickle=True)
        self.A_d = data['A_d']
        self.B_d = data['B_d']
        self.Hx = data['Hx']
        self.hx = data['hx']
        self.Hu = data['Hu']
        self.hu = data['hu']
        self.u_lb = data['u_lb']
        self.u_ub = data['u_ub']
        
        # Terminal cost (DARE solution for stability)
        self.P_terminal = la.solve_discrete_are(self.A_d, self.B_d, self.Q, self.R)
        
        # Build CVXPY problem
        self._build_cvxpy_problem()
    
    def _build_cvxpy_problem(self):
        """Construct the CVXPY parametric problem."""
        self.X = cp.Variable((self.nx, self.N + 1))
        self.U = cp.Variable((self.nu, self.N))
        self.x_curr_param = cp.Parameter(self.nx)
        self.x_ref_param = cp.Parameter(self.nx)
        
        # Cost function
        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(self.X[:, k] - self.x_ref_param, self.Q)
            cost += cp.quad_form(self.U[:, k], self.R)
        cost += cp.quad_form(self.X[:, self.N] - self.x_ref_param, self.P_terminal)
        
        # Constraints
        constraints = []
        
        # Initial condition (tight coupling: true state == nominal state)
        constraints.append(self.X[:, 0] == self.x_curr_param)
        
        # Dynamics
        for k in range(self.N):
            constraints.append(self.X[:, k + 1] == self.A_d @ self.X[:, k] + self.B_d @ self.U[:, k])
        
        # State constraints (original, not tightened)
        for k in range(self.N + 1):
            constraints.append(self.Hx @ self.X[:, k] <= self.hx)
        
        # Input constraints
        for k in range(self.N):
            constraints.append(self.Hu @ self.U[:, k] <= self.hu)
        
        self.problem = cp.Problem(cp.Minimize(cost), constraints)
    
    def solve(self, x0: np.ndarray, x_ref: np.ndarray) -> Dict:
        """
        Solve MPC optimization.
        
        Args:
            x0: Current state (6,) [px, py, pz, vx, vy, vz]
            x_ref: Reference state (6,) [px, py, pz, vx, vy, vz]
        
        Returns:
            Dict with 'u', 'x_traj', 'u_traj', 'status', 'solve_time'
        """
        import time
        t_start = time.time()
        
        self.x_curr_param.value = x0
        self.x_ref_param.value = x_ref
        
        self.problem.solve(solver=cp.OSQP, warm_start=True)
        
        solve_time = time.time() - t_start
        
        if self.problem.status in ["optimal", "optimal_inaccurate"] and self.X.value is not None:
            u = self.U.value[:, 0]
            u = np.clip(u, self.u_lb, self.u_ub)
            
            return {
                'u': u,
                'x_traj': self.X.value.T.copy(),
                'u_traj': self.U.value.T.copy(),
                'status': self.problem.status,
                'solve_time': solve_time,
                'trigger': True  # Always triggers (solves every step)
            }
        else:
            # Fallback: zero acceleration
            return {
                'u': np.zeros(3),
                'x_traj': np.tile(x0, (self.N + 1, 1)),
                'u_traj': np.zeros((self.N, 3)),
                'status': f"infeasible_{self.problem.status}",
                'solve_time': solve_time,
                'trigger': True
            }


class TubeMPC:
    """Tube MPC solver - solves every step with tube constraints and auxiliary controller."""
    
    def __init__(self, horizon: int = 20, dt: float = 0.05,
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None,
                 tube_data_path: Optional[str] = None):
        """
        Initialize Tube MPC solver.
        
        Args:
            horizon: MPC prediction horizon N
            dt: Sample time (seconds)
            Q: State cost matrix (6x6)
            R: Input cost matrix (3x3)
            tube_data_path: Path to tube_data.npz
        """
        self.N = horizon
        self.dt = dt
        self.nx = 6
        self.nu = 3
        
        # Default weights
        self.Q = Q if Q is not None else np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
        self.R = R if R is not None else np.diag([0.1, 0.1, 0.1])
        
        # Load tube data
        if tube_data_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            tube_data_path = str(base_dir / 'simulations' / 'tube_data.npz')
        
        data = np.load(tube_data_path, allow_pickle=True)
        self.A_d = data['A_d']
        self.B_d = data['B_d']
        self.K = data['K']  # Local feedback gain
        self.Hx_tight = data['Hx_tight']
        self.hx_tight = data['hx_tight']
        self.Hu_tight = data['Hu_tight']
        self.hu_tight = data['hu_tight']
        self.Omega_H = data['Omega_H']
        self.Omega_h = data['Omega_h']
        self.u_lb = data['u_lb']
        self.u_ub = data['u_ub']
        
        # Terminal cost
        self.P_terminal = la.solve_discrete_are(self.A_d, self.B_d, self.Q, self.R)
        
        # Build CVXPY problem
        self._build_cvxpy_problem()
        
        # Initialize nominal trajectory
        self.reset()
    
    def _build_cvxpy_problem(self):
        """Construct the CVXPY parametric problem."""
        self.X_bar = cp.Variable((self.nx, self.N + 1))
        self.U_bar = cp.Variable((self.nu, self.N))
        self.x_curr_param = cp.Parameter(self.nx)
        self.x_ref_param = cp.Parameter(self.nx)
        
        # Cost function (nominal trajectory cost)
        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(self.X_bar[:, k] - self.x_ref_param, self.Q)
            cost += cp.quad_form(self.U_bar[:, k], self.R)
        cost += cp.quad_form(self.X_bar[:, self.N] - self.x_ref_param, self.P_terminal)
        
        # Constraints
        constraints = []
        
        # Initial condition: x0 must be in tube around nominal
        constraints.append(self.Omega_H @ (self.x_curr_param - self.X_bar[:, 0]) <= self.Omega_h)
        
        # Dynamics
        for k in range(self.N):
            constraints.append(self.X_bar[:, k + 1] == self.A_d @ self.X_bar[:, k] + self.B_d @ self.U_bar[:, k])
        
        # Tightened state constraints
        for k in range(self.N + 1):
            constraints.append(self.Hx_tight @ self.X_bar[:, k] <= self.hx_tight)
        
        # Tightened input constraints
        for k in range(self.N):
            constraints.append(self.Hu_tight @ self.U_bar[:, k] <= self.hu_tight)
        
        self.problem = cp.Problem(cp.Minimize(cost), constraints)
    
    def reset(self):
        """Reset nominal trajectory memory."""
        self.X_nom_traj = np.zeros((self.nx, self.N + 1))
        self.U_nom_traj = np.zeros((self.nu, self.N))
    
    def solve(self, x0: np.ndarray, x_ref: np.ndarray) -> Dict:
        """
        Solve MPC optimization.
        
        Args:
            x0: Current state (6,)
            x_ref: Reference state (6,)
        
        Returns:
            Dict with 'u', 'x_traj', 'u_traj', 'status', 'solve_time'
        """
        import time
        t_start = time.time()
        
        # Set parameters
        self.x_curr_param.value = x0
        self.x_ref_param.value = x_ref
        
        # Warm start with previous solution
        self.X_bar.value = self.X_nom_traj
        self.U_bar.value = self.U_nom_traj
        
        # Solve
        self.problem.solve(solver=cp.OSQP, warm_start=True)
        
        solve_time = time.time() - t_start
        
        if self.problem.status in ["optimal", "optimal_inaccurate"] and self.X_bar.value is not None:
            self.X_nom_traj = self.X_bar.value.copy()
            self.U_nom_traj = self.U_bar.value.copy()
            
            # True control: nominal + feedback correction
            x_bar = self.X_nom_traj[:, 0]
            u_bar = self.U_nom_traj[:, 0]
            u_true = u_bar - self.K @ (x0 - x_bar)
            u_true = np.clip(u_true, self.u_lb, self.u_ub)
            
            # Propagate nominal trajectory
            self._propagate_nominal(x_ref)
            
            return {
                'u': u_true,
                'x_traj': self.X_nom_traj.T.copy(),
                'u_traj': self.U_nom_traj.T.copy(),
                'status': self.problem.status,
                'solve_time': solve_time,
                'trigger': True  # Always solves
            }
        else:
            # Fallback: use previous trajectory with LQR correction
            x_bar = self.X_nom_traj[:, 0]
            u_bar = self.U_nom_traj[:, 0]
            u_true = u_bar - self.K @ (x0 - x_bar)
            u_true = np.clip(u_true, self.u_lb, self.u_ub)
            
            self._propagate_nominal(x_ref)
            
            return {
                'u': u_true,
                'x_traj': self.X_nom_traj.T.copy(),
                'u_traj': self.U_nom_traj.T.copy(),
                'status': f"fallback_{self.problem.status}",
                'solve_time': solve_time,
                'trigger': True
            }
    
    def _propagate_nominal(self, x_ref: np.ndarray):
        """Roll nominal trajectory forward by one step."""
        self.X_nom_traj = np.roll(self.X_nom_traj, -1, axis=1)
        self.U_nom_traj = np.roll(self.U_nom_traj, -1, axis=1)
        
        # Pad end with LQR control
        x_end = self.X_nom_traj[:, -2]
        u_pad = -self.K @ (x_end - x_ref)
        self.U_nom_traj[:, -1] = u_pad
        self.X_nom_traj[:, -1] = self.A_d @ x_end + self.B_d @ u_pad


class ETTubeMPC(TubeMPC):
    """Event-Triggered Tube MPC solver - only solves when error exits invariant set."""
    
    def __init__(self, horizon: int = 20, dt: float = 0.05,
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None,
                 tube_data_path: Optional[str] = None,
                 trigger_threshold: float = 0.15,
                 consecutive_threshold: int = 3):
        """
        Initialize ET-Tube MPC solver.
        
        Args:
            horizon: MPC prediction horizon N
            dt: Sample time (seconds)
            Q: State cost matrix (6x6)
            R: Input cost matrix (3x3)
            tube_data_path: Path to tube_data.npz
            trigger_threshold: Position error threshold (meters)
            consecutive_threshold: Consecutive steps above threshold to trigger
        """
        super().__init__(horizon, dt, Q, R, tube_data_path)
        
        self.trigger_threshold = trigger_threshold
        self.consecutive_threshold = consecutive_threshold
        self.trigger_counter = 0
        self.steps_since_solve = 0
    
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.trigger_counter = 0
        self.steps_since_solve = self.N  # Force solve on first step
    
    def solve(self, x0: np.ndarray, x_ref: np.ndarray) -> Dict:
        """
        Solve MPC optimization (Event-Triggered).
        
        Args:
            x0: Current state (6,)
            x_ref: Reference state (6,)
        
        Returns:
            Dict with 'u', 'x_traj', 'u_traj', 'status', 'solve_time', 'trigger'
        """
        import time
        t_start = time.time()
        
        # Check event trigger condition
        x_bar = self.X_nom_traj[:, 0]
        pos_error = np.linalg.norm(x0[0:3] - x_bar[0:3])
        
        if pos_error > self.trigger_threshold:
            self.trigger_counter += 1
        else:
            self.trigger_counter = 0
        
        # Trigger if: (1) consecutive threshold met, or (2) haven't solved in too long
        should_solve = (self.trigger_counter >= self.consecutive_threshold or 
                       self.steps_since_solve >= self.N - 2)
        
        trigger = should_solve
        solve_status = "nominal_propagated"
        
        if should_solve:
            # Solve optimization
            self.x_curr_param.value = x0
            self.x_ref_param.value = x_ref
            
            self.X_bar.value = self.X_nom_traj
            self.U_bar.value = self.U_nom_traj
            
            self.problem.solve(solver=cp.OSQP, warm_start=True)
            
            if self.problem.status in ["optimal", "optimal_inaccurate"] and self.X_bar.value is not None:
                self.X_nom_traj = self.X_bar.value.copy()
                self.U_nom_traj = self.U_bar.value.copy()
                solve_status = self.problem.status
            else:
                solve_status = f"fallback_{self.problem.status}"
            
            self.trigger_counter = 0
            self.steps_since_solve = 0
        else:
            self.steps_since_solve += 1
        
        # Compute true control (always, regardless of solve)
        x_bar = self.X_nom_traj[:, 0]
        u_bar = self.U_nom_traj[:, 0]
        u_true = u_bar - self.K @ (x0 - x_bar)
        u_true = np.clip(u_true, self.u_lb, self.u_ub)
        
        # Propagate nominal trajectory
        self._propagate_nominal(x_ref)
        
        solve_time = time.time() - t_start
        
        return {
            'u': u_true,
            'x_traj': self.X_nom_traj.T.copy(),
            'u_traj': self.U_nom_traj.T.copy(),
            'status': solve_status,
            'solve_time': solve_time,
            'trigger': trigger,
            'pos_error': pos_error
        }
