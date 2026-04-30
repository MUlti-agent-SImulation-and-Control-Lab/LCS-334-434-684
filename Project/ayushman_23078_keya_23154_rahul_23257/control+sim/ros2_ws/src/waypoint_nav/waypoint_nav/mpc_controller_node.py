import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

import numpy as np
import torch
import casadi as ca
import os
import time
from datetime import datetime
from builtin_interfaces.msg import Time as RosTime

from geometry_msgs.msg import TwistStamped, PoseArray, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32MultiArray

import tf_transformations

import sys
sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    MLP, get_casadi_mlp, trunk_input_dim, branch_input_dim,
    output_dim_total, p, N_horizon, Nx, Nu, Ts, ROBOT_WIDTH,
    MAX_WHEEL_SPEED, MAX_ANGULAR_VEL, Q, R, R_delta
)
from fault_tolerence import OnlineGainEstimator

ODOM_STALENESS_LIMIT = 0.4 


class MPCControllerNode(Node):

    def __init__(self):
        super().__init__('mpc_controller_node')

        self.declare_parameter('model_path',
            '/root/ros2_ws/src/waypoint_nav/config/deepONet_model.pt')
        self.declare_parameter('waypoints_file',
            '/root/ros2_ws/src/waypoint_nav/config/waypoints.csv')
        self.declare_parameter('dist_threshold', 0.3)
        self.declare_parameter('use_gain_estimator', True)

        self.declare_parameter('fault_enabled', False)
        self.declare_parameter('fault_step', 75)
        self.declare_parameter('fault_side', 'left')
        self.declare_parameter('fault_magnitude', 0.6)

        model_path         = self.get_parameter('model_path').value
        waypoints_file     = self.get_parameter('waypoints_file').value
        self.dist_thresh   = self.get_parameter('dist_threshold').value
        self.use_estimator = self.get_parameter('use_gain_estimator').value

        self.fault_enabled   = self.get_parameter('fault_enabled').value
        self.fault_step      = self.get_parameter('fault_step').value
        self.fault_side      = self.get_parameter('fault_side').value
        self.fault_magnitude = self.get_parameter('fault_magnitude').value

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'[INIT] Loading model from: {model_path}  (device={device})')
        checkpoint = torch.load(model_path, map_location=device)

        self.trunk_net  = MLP(trunk_input_dim, p)
        self.branch_net = MLP(branch_input_dim, p * output_dim_total)
        self.trunk_net.load_state_dict(checkpoint['trunk_net'])
        self.branch_net.load_state_dict(checkpoint['branch_net'])
        self.trunk_net.eval()
        self.branch_net.eval()
        self.get_logger().info('[INIT] DeepONet model loaded successfully.')

        self._build_mpc()
        self.get_logger().info('[INIT] MPC solver (CasADi/IPOPT) built successfully.')

        if self.use_estimator:
            self.estimator = OnlineGainEstimator(self.trunk_net, self.branch_net)
            self.get_logger().info('[INIT] OnlineGainEstimator initialized.')

        self.estimated_gains = np.array([1.0, 1.0])

        self.waypoints    = None
        self.waypoint_idx = 0

        if waypoints_file:
            if waypoints_file.endswith('.csv'):
                wps = np.loadtxt(waypoints_file, delimiter=',', skiprows=1)
            else:
                wps = np.load(waypoints_file)
            wps = np.atleast_2d(wps)[:, :2]
            self.get_logger().info(f'[INIT] Waypoints loaded ({len(wps)} points):\n{wps}')
            self._set_waypoints(wps)

        self.curr_x        = None  
        self.curr_pitch    = 0.0
        self.odom_time     = None
        self.prev_state    = None
        self.prev_u        = None
        self.u_guess       = np.zeros(Nu * N_horizon)
        self.step_count    = 0

        self._solver_times = []

        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/global', self._odom_callback, 10)
        self.waypoint_sub = self.create_subscription(
            PoseArray, '/mpc/waypoints', self._waypoints_callback, 10)

        self.cmd_pub    = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/mpc/status', 10)

        self.gains_pub      = self.create_publisher(Float32MultiArray, '/mpc/gains', 10)
        self.wheel_cmds_pub = self.create_publisher(Float32MultiArray, '/mpc/wheel_cmds', 10)
        self.errors_pub     = self.create_publisher(Float32MultiArray, '/mpc/errors', 10)

        self.timer = self.create_timer(Ts, self._control_loop)
        self.get_logger().info('[INIT] MPCControllerNode ready. Waiting for odometry and waypoints...')


    def _build_mpc(self):
        opti = ca.Opti()

        U_seq               = opti.variable(Nu * N_horizon)
        current_state_param = opti.parameter(Nx)
        ref_traj_param      = opti.parameter(Nx * N_horizon)
        gains_param         = opti.parameter(2)

        U_reshaped   = ca.reshape(U_seq, (Nu, N_horizon)).T
        U_scaled_mat = U_reshaped * ca.repmat(gains_param.T, N_horizon, 1)
        U_scaled     = ca.reshape(U_scaled_mat.T, (Nu * N_horizon, 1))

        theta_curr      = current_state_param[2]
        trunk_input_sym = ca.vertcat(ca.cos(theta_curr), ca.sin(theta_curr))

        t_sym      = get_casadi_mlp(self.trunk_net, trunk_input_sym)
        b_sym      = get_casadi_mlp(self.branch_net, U_scaled)
        b_reshaped = ca.reshape(b_sym, (p, output_dim_total)).T
        Y_delta_pred = ca.mtimes(b_reshaped, t_sym) 

        obj = 0
        for k in range(N_horizon):
            idx = k * Nx
            weight_multiplier = 5.0 if k == N_horizon - 1 else 1.0

            x_pred  = current_state_param[0] + Y_delta_pred[idx]
            y_pred  = current_state_param[1] + Y_delta_pred[idx + 1]
            th_pred = current_state_param[2] + Y_delta_pred[idx + 2]

            x_ref  = ref_traj_param[idx]
            y_ref  = ref_traj_param[idx + 1]
            th_ref = ref_traj_param[idx + 2]

            th_err         = th_pred - th_ref
            th_err_wrapped = ca.atan2(ca.sin(th_err), ca.cos(th_err))

            state_err = ca.vertcat(x_pred - x_ref, y_pred - y_ref, th_err_wrapped)
            u_k       = U_seq[k * Nu : (k + 1) * Nu]

            obj += weight_multiplier * ca.mtimes([state_err.T, Q, state_err])
            obj += ca.mtimes([u_k.T, R, u_k])

            if k > 0:
                du   = u_k - U_seq[(k - 1) * Nu : k * Nu]
                obj += ca.mtimes([du.T, R_delta, du])

        opti.minimize(obj)

        for k in range(N_horizon):
            vl_k = U_seq[k * Nu]
            vr_k = U_seq[k * Nu + 1]
            opti.subject_to(opti.bounded(-MAX_WHEEL_SPEED, vl_k, MAX_WHEEL_SPEED))
            opti.subject_to(opti.bounded(-MAX_WHEEL_SPEED, vr_k, MAX_WHEEL_SPEED))
            omega_k = (vr_k - vl_k) / ROBOT_WIDTH
            opti.subject_to(opti.bounded(-MAX_ANGULAR_VEL, omega_k, MAX_ANGULAR_VEL))

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 200,
        }
        opti.solver('ipopt', opts)

        self.opti                = opti
        self.U_seq               = U_seq
        self.current_state_param = current_state_param
        self.ref_traj_param      = ref_traj_param
        self.gains_param         = gains_param


    def _build_reference(self, state: np.ndarray, target: np.ndarray) -> np.ndarray:
        refs = []
        cx, cy, cth = state[0], state[1], state[2]
        gx, gy = target[0], target[1]

        total_dist     = np.hypot(gx - cx, gy - cy)
        lookahead_dist = min(total_dist, 1.5)
        angle_to_goal  = np.arctan2(gy - cy, gx - cx)

        for k in range(1, N_horizon + 1):
            step_dist = lookahead_dist * (k / N_horizon)
            rx = cx + step_dist * np.cos(angle_to_goal)
            ry = cy + step_dist * np.sin(angle_to_goal)
            rth = cth if total_dist < 0.4 else angle_to_goal
            refs.extend([rx, ry, rth])
        return np.array(refs)


    def _odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, pitch, yaw = tf_transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w])
        self.curr_pitch = pitch
        self.curr_x     = np.array([x, y, yaw])
        self.odom_time  = self.get_clock().now()

    def _waypoints_callback(self, msg: PoseArray):
        wps = np.array([[p.position.x, p.position.y] for p in msg.poses])
        self._set_waypoints(wps)
        self.get_logger().info(
            f'[WP] Received {len(self.waypoints)} waypoints via /mpc/waypoints topic.')

    def _set_waypoints(self, wps: np.ndarray):
        self.waypoints    = wps
        self.waypoint_idx = 0

    def _control_loop(self):
        if self.curr_x is None or self.waypoints is None:
            self.get_logger().info('[WAIT] Waiting for odometry and waypoints...')
            return

        wall_time_s = time.monotonic()
        state  = self.curr_x.copy()
        target = self.waypoints[self.waypoint_idx]

        if self.use_estimator and self.prev_state is not None and self.prev_u is not None:
            new_gains = self.estimator.estimate(
                u_applied=np.array(self.prev_u),
                state_prev=self.prev_state,
                state_curr=state,
            )
            alpha = 0.2
            self.estimated_gains = (alpha * new_gains
                                    + (1.0 - alpha) * self.estimated_gains)
            self.get_logger().info(
                f'[GAINS] step={self.step_count:04d}  '
                f'gain_L={self.estimated_gains[0]:.4f}  '
                f'gain_R={self.estimated_gains[1]:.4f}  '
                f'(raw_L={new_gains[0]:.4f}  raw_R={new_gains[1]:.4f})'
            )

        # Publish gains for bag recording
        gains_msg      = Float32MultiArray()
        gains_msg.data = [float(self.estimated_gains[0]),
                          float(self.estimated_gains[1])]
        self.gains_pub.publish(gains_msg)

        dx_err = target[0] - state[0]
        dy_err = target[1] - state[1]
        dist   = np.hypot(dx_err, dy_err)

        target_yaw = np.arctan2(dy_err, dx_err)
        dth_err    = target_yaw - state[2]
        dth_err    = (dth_err + np.pi) % (2 * np.pi) - np.pi  # wrap [-pi, pi]

        self.get_logger().info(
            f'[STATE] step={self.step_count:04d}  '
            f'x={state[0]:.3f}  y={state[1]:.3f}  '
            f'yaw={np.degrees(state[2]):.2f}deg  '
            f'pitch={np.degrees(self.curr_pitch):.2f}deg'
        )
        self.get_logger().info(
            f'[ERROR] step={self.step_count:04d}  '
            f'wp={self.waypoint_idx}  '
            f'dist={dist:.4f}m  '
            f'dx={dx_err:.4f}m  dy={dy_err:.4f}m  '
            f'heading_err={np.degrees(dth_err):.2f}deg'
        )

        err_msg      = Float32MultiArray()
        err_msg.data = [float(dist), float(dx_err), float(dy_err),
                        float(np.degrees(dth_err))]
        self.errors_pub.publish(err_msg)

        if dist < self.dist_thresh:
            if self.waypoint_idx < len(self.waypoints) - 1:
                self.get_logger().info(
                    f'[WP] Reached WP {self.waypoint_idx} '
                    f'(dist={dist:.4f}m < thresh={self.dist_thresh}m). '
                    f'Advancing to WP {self.waypoint_idx + 1}.'
                )
                self.waypoint_idx += 1
                return
            else:
                self.get_logger().info(
                    f'[WP] Final waypoint {self.waypoint_idx} reached '
                    f'(dist={dist:.4f}m). Sending STOP.'
                )
                self._send_stop()
                self._write_csv_row(
                    wall_time_s, state, dist, dx_err, dy_err, dth_err,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    solver_ok=True, solver_time_ms=0.0
                )
                return

        refs = self._build_reference(state, target)
        self.opti.set_value(self.current_state_param, state)
        self.opti.set_value(self.ref_traj_param, refs)
        self.opti.set_value(self.gains_param, self.estimated_gains)
        self.opti.set_initial(self.U_seq, self.u_guess)

        solver_ok      = False
        solver_time_ms = 0.0
        vl_cmd         = 0.0
        vr_cmd         = 0.0

        t_solve_start = time.monotonic()
        try:
            sol        = self.opti.solve()
            u_optimal  = sol.value(self.U_seq)
            self.u_guess = np.concatenate((u_optimal[Nu:], u_optimal[-Nu:]))
            vl_cmd, vr_cmd = u_optimal[0], u_optimal[1]
            solver_ok  = True
        except Exception as e:
            self.get_logger().warn(
                f'[SOLVER] FAILED at step={self.step_count:04d}: {e}')

        solver_time_ms = (time.monotonic() - t_solve_start) * 1000.0
        self._solver_times.append(solver_time_ms)

        self.get_logger().info(
            f'[SOLVER] step={self.step_count:04d}  '
            f'ok={solver_ok}  '
            f'time={solver_time_ms:.1f}ms  '
            f'vl_cmd={vl_cmd:.4f}m/s  '
            f'vr_cmd={vr_cmd:.4f}m/s'
        )
        wheel_msg      = Float32MultiArray()
        wheel_msg.data = [float(vl_cmd), float(vr_cmd)]
        self.wheel_cmds_pub.publish(wheel_msg)

        v_lin = (vr_cmd + vl_cmd) / 2.0
        v_ang = (vr_cmd - vl_cmd) / ROBOT_WIDTH

        K_gravity           = 0.8
        gravity_comp        = K_gravity * np.sin(self.curr_pitch)
        v_lin              += gravity_comp

        dynamic_min_v = 0.12 + 0.2 * abs(np.sin(self.curr_pitch))
        if dist > self.dist_thresh and abs(v_lin) < dynamic_min_v:
            v_lin = (np.sign(v_lin) * dynamic_min_v
                     if v_lin != 0 else dynamic_min_v)

        self.get_logger().info(
            f'[CMD] step={self.step_count:04d}  '
            f'v_lin={v_lin:.4f}m/s  '
            f'v_ang={v_ang:.4f}rad/s  '
            f'gravity_comp={gravity_comp:.4f}m/s  '
            f'dynamic_min_v={dynamic_min_v:.4f}m/s  '
            f'pitch={np.degrees(self.curr_pitch):.2f}deg'
        )

        twist_msg                  = TwistStamped()
        twist_msg.header.stamp     = self.get_clock().now().to_msg()
        twist_msg.twist.linear.x   = v_lin
        twist_msg.twist.angular.z  = v_ang
        self.cmd_pub.publish(twist_msg)

        avg_solver_ms = (np.mean(self._solver_times[-20:])
                         if self._solver_times else 0.0)
        status_str = (
            f"step:{self.step_count:04d} | "
            f"WP:{self.waypoint_idx} | "
            f"pos:({state[0]:.2f},{state[1]:.2f}) yaw:{np.degrees(state[2]):.1f}deg | "
            f"dist:{dist:.3f}m | "
            f"dx:{dx_err:.2f} dy:{dy_err:.2f} dth:{np.degrees(dth_err):.1f}deg | "
            f"vl:{vl_cmd:.3f} vr:{vr_cmd:.3f} | "
            f"v:{v_lin:.3f}m/s w:{v_ang:.3f}rad/s | "
            f"gains:L={self.estimated_gains[0]:.3f} R={self.estimated_gains[1]:.3f} | "
            f"gcomp:{gravity_comp:.3f} pitch:{np.degrees(self.curr_pitch):.1f}deg | "
            f"solver:{'OK' if solver_ok else 'FAIL'} {solver_time_ms:.0f}ms(avg:{avg_solver_ms:.0f}ms)"
        )
        status_msg      = String()
        status_msg.data = status_str
        self.status_pub.publish(status_msg)
        self.get_logger().info(f'[STATUS] {status_str}')


        self.prev_state = state
        self.prev_u     = [vl_cmd, vr_cmd]
        self.step_count += 1


    def _send_stop(self):
        try:
            stop_msg                    = TwistStamped()
            stop_msg.header.stamp       = self.get_clock().now().to_msg()
            stop_msg.header.frame_id    = 'base_link'
            stop_msg.twist.linear.x     = 0.0
            stop_msg.twist.angular.z    = 0.0
            self.cmd_pub.publish(stop_msg)
            self.get_logger().info('[CMD] STOP command published.')
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = MPCControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
      
        node._send_stop()
        rclpy.shutdown()


if __name__ == '__main__':
    main()