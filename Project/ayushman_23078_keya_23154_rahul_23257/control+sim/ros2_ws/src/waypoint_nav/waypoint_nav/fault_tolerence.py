from utils import *

class OnlineGainEstimator:
    def __init__(self, trunk_net, branch_net):
        """
        Solves a small optimization problem to find the efficiency gains
        that explain the difference between Previous State and Current State.
        """
        self.opti = ca.Opti()

        # Variables: The efficiency gains [gain_L, gain_R]
        self.gains = self.opti.variable(2)

        # Parameters: Past control and Past theta (context)
        self.u_past = self.opti.parameter(2)  # [vl, vr]
        self.theta_past = self.opti.parameter(1)

        # Parameter: The ACTUAL observed change in state (dx, dy, dth)
        self.delta_actual = self.opti.parameter(3)

        u_scaled_step = self.u_past * self.gains

        # The Branch Net expects a full horizon sequence (e.g., 20 inputs).
        # We replicate this single step to fill the vector.
        u_full_input = ca.repmat(u_scaled_step, N_horizon, 1)

        # ---------------------------------------------------------
        # 2. Network Evaluation
        # ---------------------------------------------------------
        # Branch Net: Input (20, 1) -> Output (p * 30)
        b_sym = get_casadi_mlp(branch_net, u_full_input)

        # Reshape to Matrix (30, p)
        # Assumes output_dim_total (30) and p are available from utils or context
        b_reshaped = ca.reshape(b_sym, (p, output_dim_total)).T

        # Trunk Net
        trunk_in = ca.vertcat(ca.cos(self.theta_past), ca.sin(self.theta_past))
        t_sym = get_casadi_mlp(trunk_net, trunk_in)

        # Full Prediction: (30, p) * (p, 1) -> (30, 1)
        all_pred_deltas = ca.mtimes(b_reshaped, t_sym)

        # ---------------------------------------------------------
        # 3. Output Slicing (Fixing Output Dimension Error)
        # ---------------------------------------------------------
        # We only have ground truth for the FIRST step (t0 -> t1).
        # So we extract the first 3 elements (dx, dy, dth) from the prediction.
        pred_delta_step1 = all_pred_deltas[0:Nx]  # Slices indices 0, 1, 2

        # ---------------------------------------------------------
        # 4. Optimization Objective
        # ---------------------------------------------------------
        # Now both are (3, 1)
        error = pred_delta_step1 - self.delta_actual

        # Cost: Squared Error + Regularization (prefer gains close to 1.0)
        self.opti.minimize(ca.mtimes(error.T, error) + 10 * ca.sumsqr(self.gains - 1.0))

        # Constraints: Gains must be physical (0.0 to 1.25)
        self.opti.subject_to(self.opti.bounded(0.0, self.gains, 1.25))

        # Solver options (Fast solver needed)
        p_opts = {'expand': True}
        s_opts = {'print_level': 0, 'sb': 'yes'}
        self.opti.solver('ipopt', p_opts, s_opts)

    def estimate(self, u_applied, state_prev, state_curr):
        # Prevent division by zero or estimating on noise when stopped
        if np.linalg.norm(u_applied) < 0.01:
            return np.array([1.0, 1.0])

        dx_obs = state_curr[0] - state_prev[0]
        dy_obs = state_curr[1] - state_prev[1]
        dth_obs = state_curr[2] - state_prev[2]

        # Normalize angle wrap for dth
        dth_obs = (dth_obs + np.pi) % (2 * np.pi) - np.pi

        self.opti.set_value(self.u_past, u_applied)
        self.opti.set_value(self.theta_past, state_prev[2])
        self.opti.set_value(self.delta_actual, [dx_obs, dy_obs, dth_obs])
        self.opti.set_initial(self.gains, [1.0, 1.0])  # Warm start with healthy

        try:
            sol = self.opti.solve()
            return sol.value(self.gains)
        except:
            # If optimization fails, return default healthy gains
            return np.array([1.0, 1.0])