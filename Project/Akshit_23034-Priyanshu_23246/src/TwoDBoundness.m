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
% =========================================================================
% Paper Analytics: TP vs SS Events and UU Boundedness
% =========================================================================
clear; clc; close all;

% --- SYSTEM SETUP (Same as before) ---
A = [0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0];
B = [0 0; 1 0; 0 0; 0 1];
Q = diag([10, 1, 10, 1]); R = diag([0.1, 0.1]);     
K = lqr(A, B, Q, R); P = lyap((A - B*K)', Q);  
dt = 0.01; t_final = 10; time = 0:dt:t_final; N = length(time);
sigma = 0.05; 
steady_state_threshold = 3.0; % Define when TP ends and SS begins (seconds)

% Test 4 different starting locations to prove Uniform Ultimate Boundedness
start_points = [ [2;0;2;0], [-2;0;1;0], [0;0;-3;0], [3;0;-1;0] ];
num_tests = size(start_points, 2);

% Storage for the UUB plot
all_errors = zeros(num_tests, N);

% Storage for Event Counting (Test 1 will be our benchmark)
events_TP = 0;
events_SS = 0;

for test = 1:num_tests
    x = start_points(:, test);
    t_k = 0;
    
    for i = 1:N
        t = time(i);
        xr = [cos(t); -sin(t); sin(t); cos(t)];
        ur = [-cos(t); -sin(t)];
        ur_dot = [sin(t); -cos(t)];
        ur_ddot = [cos(t); sin(t)];
        
        e = x - xr;
        all_errors(test, i) = norm(e);
        
        if t == t_k
            % If this is Test 1, count the events for our TP/SS graph
            if test == 1
                if t < steady_state_threshold
                    events_TP = events_TP + 1;
                else
                    events_SS = events_SS + 1;
                end
            end
            
            c0 = ur - K*e;
            c1 = ur_dot - K*(A - B*K)*e;
            c2 = 0.5 * (ur_ddot - K*((A - B*K)^2)*e);
        end
        
        delta_t = t - t_k;
        u_applied = c0 + c1*delta_t + c2*(delta_t^2);
        u_ideal = ur - K*e;
        epsilon = u_applied - u_ideal;
        
        if norm(2 * e' * P * B * epsilon) >= sigma * (e' * Q * e) && t > t_k
            t_k = t + dt; 
        end
        
        x = x + (A*x + B*u_applied) * dt;
    end
end

% --- PLOTTING THE ANALYTICS ---
figure('Position', [150, 150, 1000, 400], 'Name', 'Advanced Paper Analytics');

% Plot 1: Uniform Ultimate Boundedness (UUB)
subplot(1,2,1);
hold on;
colors = ['b', 'r', 'g', 'm'];
for test = 1:num_tests
    plot(time, all_errors(test, :), colors(test), 'LineWidth', 1.2);
end
% Calculate the max error during the steady state to draw the UU Bound line
uu_bound_value = max(max(all_errors(:, int32(steady_state_threshold/dt):end)));
yline(uu_bound_value, 'k--', 'LineWidth', 2, 'Label', sprintf('UU Bound: %.4f', uu_bound_value));
title('Uniform Ultimate Boundedness (Multiple Paths)');
xlabel('Time (s)'); ylabel('Tracking Error Norm ||e||');
legend('Start (2,2)', 'Start (-2,1)', 'Start (0,-3)', 'Start (3,-1)');
grid on;

% Plot 2: TP vs SS Event Density
subplot(1,2,2);
bar_labels = categorical({'Transient Phase (0-3s)', 'Steady State (3-10s)'});
bar_data = [events_TP, events_SS];
b = bar(bar_labels, bar_data, 0.5);
b.FaceColor = 'flat';
b.CData(1,:) = [0.8500 0.3250 0.0980]; % Orange for TP
b.CData(2,:) = [0 0.4470 0.7410];      % Blue for SS
title('Event Distribution (TP vs SS)');
ylabel('Number of Transmissions');
% Add numbers on top of bars
xtips = b(1).XEndPoints;
ytips = b(1).YEndPoints;
labels = string(b(1).YData);
text(xtips, ytips, labels, 'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
