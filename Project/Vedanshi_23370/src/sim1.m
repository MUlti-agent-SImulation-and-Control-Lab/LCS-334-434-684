clc; clear; close all;

%% Parameters
N       = 4;
T_final = 100;
dt      = 0.001;
tvec    = 0:dt:T_final;
steps   = length(tvec);
gamma1  = 1.0;
gamma2  = 1.0;

%% Graph (square)
k = 0.5;
A = [0 1 0 1;
     1 0 1 0;
     0 1 0 1;
     1 0 1 0];
L = diag(sum(A,2)) - A;

%% Initial states
x = k * [ 1  2;
         -2  1;
          3 -4;
         -2  1];
v     = zeros(N,2);
alpha = zeros(N,1);

%% Cost gradients (as function handles)
t1 = [2; 1];
Q2 = [3 0; 0 1];
t4 = [-2; 2];

grads = {
    @(x) 2*(x - t1);
    @(x) 2*(Q2 * x);
    @(x) [cos(x(1)); 2*(x(2)-1)];
    @(x) 2*(x - t4)
};

%% Storage
X_hist = zeros(N,2,steps);

%% Simulation
for t = 1:steps
    X_hist(:,:,t) = x;

    [x_dot, v_dot, alpha_dot] = dyn_adaptive(x, v, alpha, L, grads, gamma1, gamma2);

    x     = x + dt * x_dot;
    v     = v + dt * v_dot;
    alpha = alpha + dt * alpha_dot;
end

disp('Simulation done.');
disp('Final positions:');
disp(X_hist(:,:,end));

%% Plot 1: Pairwise disagreement
pairs = [1 2; 2 3; 3 4; 4 1; 1 3; 2 4];

figure; hold on;
for p = 1:size(pairs,1)
    i = pairs(p,1);
    j = pairs(p,2);

    diff = squeeze(X_hist(i,:,:) - X_hist(j,:,:));
    dij  = sum(diff.^2,1);
    plot(tvec, dij, 'LineWidth',1.5);
end
xlabel('Time');
ylabel('||x_i - x_j||^2');
title('Pairwise Disagreement');
grid on;

%% Plot 2: Components
figure;
subplot(1,2,1); hold on;
subplot(1,2,2); hold on;

for i = 1:N
    subplot(1,2,1);
    plot(tvec, squeeze(X_hist(i,1,:)));

    subplot(1,2,2);
    plot(tvec, squeeze(X_hist(i,2,:)));
end

subplot(1,2,1); title('x1'); xlabel('Time'); grid on;
subplot(1,2,2); title('x2'); xlabel('Time'); grid on;

%% Plot 3: Trajectories
figure; hold on;
colors = lines(N);

for i = 1:N
    traj = squeeze(X_hist(i,:,:))';
    plot(traj(:,1), traj(:,2), 'Color', colors(i,:));
    plot(traj(1,1), traj(1,2), 'o', 'Color', colors(i,:));
    plot(traj(end,1), traj(end,2), 's', 'Color', colors(i,:));
end

xlabel('x_1'); ylabel('x_2');
title('2D Trajectories');
axis equal; grid on;

%% ── Dynamics Function ───────────────────────────────────────────────────────
function [x_dot, v_dot, alpha_dot] = dyn_adaptive(x, v, alpha, L, grads, gamma1, gamma2)

N = size(x,1);

x_dot     = zeros(size(x));
v_dot     = zeros(size(v));
alpha_dot = zeros(N,1);

Lx = L * x;

for i = 1:N
    xi = x(i,:)';   % column
    vi = v(i,:)';

    g      = grads{i}(xi);
    e_i    = Lx(i,:)';
    beta_i = e_i' * e_i;
    gain   = alpha(i) + beta_i;

    v_dot(i,:)     = (gamma1 * gain * e_i)';
    x_dot(i,:)     = (-gamma2 * g - gamma1 * gain * e_i - vi)';
    alpha_dot(i)   = beta_i;
end
end