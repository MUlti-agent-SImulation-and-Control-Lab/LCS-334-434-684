% =========================================================================
% 3D Event-Triggered Polynomial Control (Drone Flight Path)
% =========================================================================
clear; clc; close all;

% --- 1. 3D SYSTEM DYNAMICS & LQR ---
% 6 States: X-pos, X-vel, Y-pos, Y-vel, Z-pos, Z-vel
A = zeros(6,6);
A(1,2) = 1; A(3,4) = 1; A(5,6) = 1;

% 3 Inputs: X-accel, Y-accel, Z-accel
B = zeros(6,3);
B(2,1) = 1; B(4,2) = 1; B(6,3) = 1;

% Q penalizes positions (10) and velocities (1) for X, Y, Z
Q = diag([10, 1, 10, 1, 10, 1]); 
% R penalizes acceleration effort for X, Y, Z
R = diag([0.1, 0.1, 0.1]);     

K = lqr(A, B, Q, R);      % Calculate 3x6 Gain Matrix
P = lyap((A - B*K)', Q);  % Lyapunov matrix for the trigger

% --- 2. SIMULATION SETUP ---
dt = 0.01;                
t_final = 15;             
time = 0:dt:t_final;
N = length(time);

% Start off-path at (x=2, y=2, z=2)
x = [2; 0; 2; 0; 2; 0];         
t_k = 0;                  
sigma = 0.05;             

% Pre-compute matrix powers for polynomial
A_cl = A - B*K;
A_cl_2 = A_cl * A_cl;

% Storage
x_hist = zeros(6, N);
xr_hist = zeros(6, N);
event_times = [];

% --- 3. MAIN LOOP ---
for i = 1:N
    t = time(i);
    
    % a) Define 3D Trajectory (Lissajous Knot)
    % xr = [px; vx; py; vy; pz; vz]
    xr = [sin(t); cos(t); sin(2*t); 2*cos(2*t); cos(t); -sin(t)];
    
    % ur = [ax; ay; az]
    ur = [-sin(t); -4*sin(2*t); -cos(t)];
    ur_dot = [-cos(t); -8*cos(2*t); sin(t)];
    ur_ddot = [sin(t); 16*sin(2*t); cos(t)];
    
    % Exact error
    e = x - xr;
    
    % b) Calculate Polynomial Coefficients (Now 3x1 Vectors!)
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
        t_k = t + dt; 
    end
    
    % e) Update Physics
    x = x + (A*x + B*u_applied) * dt;
    
    % Save data
    x_hist(:, i) = x;
    xr_hist(:, i) = xr;
end

% --- 4. PLOT RESULTS ---
figure('Position', [100, 100, 900, 700]);

% Plot 1: 3D Trajectory
subplot(2,1,1);
plot3(xr_hist(1,:), xr_hist(3,:), xr_hist(5,:), 'k--', 'LineWidth', 2); hold on;
plot3(x_hist(1,:), x_hist(3,:), x_hist(5,:), 'b', 'LineWidth', 2);
scatter3(x_hist(1,1), x_hist(3,1), x_hist(5,1), 70, 'ro', 'filled'); % Start point
title('3D Trajectory Tracking (Drone Flight Path)');
xlabel('X Position'); ylabel('Y Position'); zlabel('Z Position (Altitude)');
legend('Reference 3D Path', 'Drone Path', 'Start (2,2,2)'); 
grid on; axis equal; view(45, 30); % Set a nice 3D viewing angle

% Plot 2: Event Times
subplot(2,1,2);
stem(event_times(1:end-1), diff(event_times), 'filled');
title('Time Between Communication Events (3D System)');
xlabel('Time (s)'); ylabel('Seconds between packets'); 
grid on;

fprintf('Events triggered: %d out of %d steps.\n', length(event_times), N);