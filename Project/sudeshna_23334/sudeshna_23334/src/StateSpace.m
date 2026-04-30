%% StateSpace.m  –  2D LIPM-MPC for bipedal walking
% track ZMP staircase reference

%% 1. Physical Constants
g    = 9.81;
zc   = 0.90;    % CoM height (m)

%% 2. 1D LIPM Building Block
%  State : [pos; vel; zmp]    Input: zmp_dot (rate of ZMP change)
A_1d = [0,    1,    0;
        g/zc, 0,   -g/zc;
        0,    0,    0];
B_1d = [0; 0; 1];
%  Output: [CoM position; ZMP position]
C_1d = [1, 0, 0;
        0, 0, 1];
D_1d = [0; 0];

%% 3. 2D Combined System
%  State : [x; xd; zmp_x;  y; yd; zmp_y]   (6 states)
%  Input : [zmp_x_dot; zmp_y_dot]           (2 inputs)
%  Output: [CoM_x; ZMP_x; CoM_y; ZMP_y]    (4 outputs)
A = blkdiag(A_1d, A_1d);
B = blkdiag(B_1d, B_1d);
C = blkdiag(C_1d, C_1d);
D = zeros(4,2);

lipm_sys = ss(A, B, C, D);

%% 4. Sample Time & MPC Object
Ts = 0.05;
mpc_controller = mpc(lipm_sys, Ts);

%% 5. Horizons
mpc_controller.PredictionHorizon = 30;   % 1.5 s lookahead
mpc_controller.ControlHorizon    = 8;

%% 6. Output Weights  [CoM_x, ZMP_x, CoM_y, ZMP_y]
%  Track ZMP reference tightly, allow CoM to move naturally
mpc_controller.Weights.OutputVariables = [2, 20, 2, 20];

%% 7. Input Rate Weights  [sagittal, lateral]
%  Low rate weight = controller can shift ZMP quickly between feet
mpc_controller.Weights.ManipulatedVariablesRate = [0.05, 0.05];

%% 8. ZMP Velocity Limits
mpc_controller.ManipulatedVariables(1).Min = -3.0;
mpc_controller.ManipulatedVariables(1).Max =  3.0;
mpc_controller.ManipulatedVariables(2).Min = -3.0;
mpc_controller.ManipulatedVariables(2).Max =  3.0;

%% 9. ZMP Position Constraints  (support polygon)
lx            = 0.20;   % foot length sagittal
ly             = 0.12;  % foot width  lateral
safety_margin = 0.02;

% Sagittal ZMP must stay within foot
mpc_controller.OutputVariables(2).Min = -(lx/2 - safety_margin);
mpc_controller.OutputVariables(2).Max =  (lx/2 - safety_margin);

% Lateral ZMP – wider window during step transition
mpc_controller.OutputVariables(4).Min = -(ly/2 + 0.04);  % allow stepping over
mpc_controller.OutputVariables(4).Max =  (ly/2 + 0.04);

%% 10. Initial State
x0 = zeros(6,1);

%% 11. Report
fprintf('============================================\n');
fprintf('  2D LIPM-MPC (Walking) Initialised\n');
fprintf('  Ts            = %.3f s\n', Ts);
fprintf('  CoM height zc = %.3f m\n', zc);
fprintf('  Pred Horizon  = %d steps (%.2f s)\n', ...
         mpc_controller.PredictionHorizon, ...
         mpc_controller.PredictionHorizon*Ts);
fprintf('  ZMP X limits  = [%.3f, %.3f] m\n', ...
         mpc_controller.OutputVariables(2).Min, ...
         mpc_controller.OutputVariables(2).Max);
fprintf('  ZMP Y limits  = [%.3f, %.3f] m\n', ...
         mpc_controller.OutputVariables(4).Min, ...
         mpc_controller.OutputVariables(4).Max);
fprintf('============================================\n');
