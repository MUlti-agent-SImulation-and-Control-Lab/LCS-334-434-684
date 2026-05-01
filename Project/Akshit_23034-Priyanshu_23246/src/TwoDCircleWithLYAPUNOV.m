% =========================================================================
% Event-Triggered Polynomial Control (p=2) for 2D Circular Tracking
% =========================================================================
clear; clc; close all;

% --- 1. INITIALIZE SYSTEM DYNAMICS ---
A = [0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0];
B = [0 0; 1 0; 0 0; 0 1];

% --- 2. CALCULATE LQR GAIN (K) AND LYAPUNOV MATRIX (P) ---
Q = diag([10, 1, 10, 1]); 
R = diag([0.1, 0.1]);     
K = lqr(A, B, Q, R);      
P = lyap((A - B*K)', Q);  

% --- 3. SIMULATION PARAMETERS ---
dt = 0.01;                  % Simulation time step
t_final = 15;               % Total simulation time
time = 0:dt:t_final;
N = length(time);

x = [2; 0; 2; 0];           % Start off-path to prove stability
t_k = 0;                    % Time of the last event
sigma = 0.05;               % Trigger threshold (5%)

% Precompute closed-loop matrices for the Taylor expansion
A_cl = A - B*K; 
A_cl_2 = A_cl * A_cl;

% Data storage for plotting
x_hist = zeros(4, N);
xr_hist = zeros(4, N);
u_app_hist = zeros(2, N);
error_norm_hist = zeros(1, N);
event_times = [];

% --- 4. MAIN SIMULATION LOOP ---
for i = 1:N
    t = time(i);
    
    % a) Generate Reference Trajectory (Circle)
    xr = [cos(t); -sin(t); sin(t); cos(t)];
    ur = [-cos(t); -sin(t)];
    ur_dot = [sin(t); -cos(t)];
    ur_ddot = [cos(t); sin(t)];
    
    % Calculate actual tracking error
    e = x - xr;
    error_norm_hist(i) = norm(e);
    
    % b) Network Event: Calculate new polynomial coefficients
    if t == t_k
        event_times = [event_times, t]; 
        c0 = ur - K*e;
        c1 = ur_dot - K * A_cl * e;
        c2 = 0.5 * (ur_ddot - K * A_cl_2 * e);
    end
    
    % c) Onboard Computer: Apply polynomial control
    delta_t = t - t_k;
    u_applied = c0 + c1*delta_t + c2*(delta_t^2);
    u_app_hist(:, i) = u_applied;
    
    % d) Trigger Check: Monitor Lyapunov Boundary
    u_ideal = ur - K*e;
    epsilon = u_applied - u_ideal;
    
    trigger_left = norm(2 * e' * P * B * epsilon);
    trigger_right = sigma * (e' * Q * e);
    
    % If boundary is breached, schedule an event for the next time step
    if trigger_left >= trigger_right && t > t_k
        t_k = t + dt; 
    end
    
    % e) Update Physical State (Euler Integration)
    x = x + (A*x + B*u_applied) * dt;
    
    % Save data
    x_hist(:, i) = x;
    xr_hist(:, i) = xr;
end

% --- 5. GENERATE PAPER-STYLE PLOTS ---
figure('Position', [50, 50, 1100, 800], 'Name', 'Simulation Results');

% Plot A: Spatial Path
subplot(2,2,1);
plot(xr_hist(1,:), xr_hist(3,:), 'k--', 'LineWidth', 1.5); hold on;
plot(x_hist(1,:), x_hist(3,:), 'b', 'LineWidth', 1.5);
scatter(x_hist(1,1), x_hist(3,1), 50, 'ro', 'filled');
title('A. 2D Trajectory Tracking'); xlabel('X Position'); ylabel('Y Position');
legend('Reference Path', 'Robot Trajectory', 'Start Point'); grid on; axis equal;

% Plot B: Tracking Error Norm
subplot(2,2,2);
plot(time, error_norm_hist, 'r', 'LineWidth', 1.5);
title('B. Tracking Error Norm (||e(t)||)'); xlabel('Time (s)'); ylabel('Error Magnitude');
grid on;

% Plot C: Applied Control Signal
subplot(2,2,3);
plot(time, u_app_hist(1,:), 'b', 'LineWidth', 1.5); hold on;
plot(time, u_app_hist(2,:), 'm', 'LineWidth', 1.5);
title('C. ETPC Control Signals (u_{applied})'); xlabel('Time (s)'); ylabel('Acceleration Command');
legend('u_x (X-Axis)', 'u_y (Y-Axis)'); grid on;

% Plot D: Inter-Event Times
subplot(2,2,4);
inter_event_times = diff(event_times);
stem(event_times(1:end-1), inter_event_times, 'k', 'filled', 'MarkerSize', 5);
title('D. Inter-Event Times (ETR)'); xlabel('Time (s)'); ylabel('Time Between Events (s)');
ylim([0, max(inter_event_times)*1.2]); grid on;
subtitle(sprintf('Total Transmissions: %d', length(event_times)));
