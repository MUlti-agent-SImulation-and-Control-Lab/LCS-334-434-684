% =========================================================================
% 1D Event-Triggered Polynomial Control (Cosine Tracking)
% =========================================================================
clear; clc; close all;

% --- 1. SYSTEM DYNAMICS & LQR ---
A = [0 1; 0 0];
B = [0; 1];
Q = [10 0; 0 1];          % Penalize position error more than velocity
R = 0.1;                  % Penalize control effort
K = lqr(A, B, Q, R);      % Calculate 1x2 Gain Matrix
P = lyap((A - B*K)', Q);  % Lyapunov matrix for the trigger

% --- 2. SIMULATION SETUP ---
dt = 0.01;                % 10ms time step
t_final = 10;             % Simulate 10 seconds
time = 0:dt:t_final;
N = length(time);

x = [2; 0];               % Start off-path at position = 2
t_k = 0;                  % First event at t=0
sigma = 0.05;             % Strictness of event trigger

% Pre-compute matrix powers for polynomial
A_cl = A - B*K;
A_cl_2 = A_cl * A_cl;

% Storage for plotting
x_hist = zeros(2, N);
xr_hist = zeros(2, N);
event_times = [];

% --- 3. MAIN LOOP ---
for i = 1:N
    t = time(i);
    
    % a) Define Reference Trajectory (Cosine)
    xr = [cos(t); -sin(t)];
    ur = -cos(t);
    ur_dot = sin(t);
    ur_ddot = cos(t);
    
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
figure;
subplot(2,1,1);
plot(time, xr_hist(1,:), 'k--', 'LineWidth', 2); hold on;
plot(time, x_hist(1,:), 'b', 'LineWidth', 2);
title('1D Position Tracking (Cosine Wave)');
xlabel('Time (s)'); ylabel('Position');
legend('Reference \cos(t)', 'Robot Position'); grid on;

subplot(2,1,2);
stem(event_times(1:end-1), diff(event_times), 'filled');
title('Time Between Communication Events');
xlabel('Time (s)'); ylabel('Seconds between packets'); grid on;

fprintf('Events triggered: %d out of %d steps.\n', length(event_times), N);
