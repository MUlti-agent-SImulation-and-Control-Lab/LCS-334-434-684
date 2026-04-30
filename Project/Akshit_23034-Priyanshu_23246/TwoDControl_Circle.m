% =========================================================================
% 2D Event-Triggered Polynomial Control (Circle Tracking)
% =========================================================================
clear; clc; close all;

% --- 1. 2D SYSTEM DYNAMICS & LQR ---
A = [0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0];
B = [0 0; 1 0; 0 0; 0 1];

% Q penalizes X-pos, X-vel, Y-pos, Y-vel
Q = diag([10, 1, 10, 1]); 
% R penalizes X-accel, Y-accel effort
R = diag([0.1, 0.1]);     

K = lqr(A, B, Q, R);      % Calculate 2x4 Gain Matrix
P = lyap((A - B*K)', Q);  % Lyapunov matrix for the trigger

% --- 2. SIMULATION SETUP ---
dt = 0.01;                % 10ms time step
t_final = 10;             % Simulate 10 seconds
time = 0:dt:t_final;
N = length(time);

% Start off-path at (x=2, y=2) and zero velocity
x = [2; 0; 2; 0];         
t_k = 0;                  % First event at t=0
sigma = 0.05;             % Strictness of event trigger

% Pre-compute matrix powers for polynomial
A_cl = A - B*K;
A_cl_2 = A_cl * A_cl;

% Storage for plotting
x_hist = zeros(4, N);
xr_hist = zeros(4, N);
event_times = [];

% --- 3. MAIN LOOP ---
for i = 1:N
    t = time(i);
    
    % a) Define 2D Reference Trajectory (Circle)
    xr = [cos(t); -sin(t); sin(t); cos(t)];
    ur = [-cos(t); -sin(t)];
    ur_dot = [sin(t); -cos(t)];
    ur_ddot = [cos(t); sin(t)];
    
    % Exact error
    e = x - xr;
    
    % b) Calculate Polynomial Coefficients at Event Time
    if t == t_k
        event_times = [event_times, t]; 
        c0 = ur - K*e;
        c1 = ur_dot - K * A_cl * e;
        c2 = 0.5 * (ur_ddot - K * A_cl_2 * e);
    end
    
    % c) Actuator Applies Polynomial Control
    delta_t = t - t_k;
    u_applied = c0 + c1*delta_t + c2*(delta_t^2);
    
    % d) Lyapunov Event Trigger Check
    u_ideal = ur - K*e;
    epsilon = u_applied - u_ideal;
    
    trigger_left = norm(2 * e' * P * B * epsilon);
    trigger_right = sigma * (e' * Q * e);
    
    if trigger_left >= trigger_right && t > t_k
        t_k = t + dt; % Trigger next event!
    end
    
    % e) Update Physics
    x = x + (A*x + B*u_applied) * dt;
    
    % Save data
    x_hist(:, i) = x;
    xr_hist(:, i) = xr;
end

% --- 4. PLOT RESULTS ---
figure('Position', [100, 100, 800, 600]);

% Plot 1: 2D Map Trajectory
subplot(2,1,1);
plot(xr_hist(1,:), xr_hist(3,:), 'k--', 'LineWidth', 2); hold on;
plot(x_hist(1,:), x_hist(3,:), 'b', 'LineWidth', 2);
scatter(x_hist(1,1), x_hist(3,1), 50, 'ro', 'filled'); % Start point
title('2D Trajectory Tracking (Circle)');
xlabel('X Position'); ylabel('Y Position');
legend('Reference Path', 'Robot Path', 'Start (2,2)'); 
grid on; axis equal;

% Plot 2: Event Times
subplot(2,1,2);
stem(event_times(1:end-1), diff(event_times), 'filled');
title('Time Between Communication Events');
xlabel('Time (s)'); ylabel('Seconds between packets'); 
grid on;

fprintf('Events triggered: %d out of %d steps.\n', length(event_times), N);