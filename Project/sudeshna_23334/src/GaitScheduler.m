%% GaitScheduler.m  –  Joint angle trajectory from footstep pattern
%  Outputs: joint_ref timeseries for Simulink joint angle inputs
%  Joint order (8 joints): 
%    [hip_roll_L, hip_pitch_L, knee_L, ankle_L,
%     hip_roll_R, hip_pitch_R, knee_R, ankle_R]

Ts       = 0.05;
t        = (0:Ts:10)';
N        = length(t);

%% --- Gait parameters ---
step_dur  = 0.5;     % s per step
t_start   = 0.5;     % standing still before walk
swing_frac= 0.2;     % fraction of step_dur in swing phase

%% --- Standing posture (slight knee bend for stability) ---
%  All in radians
%% --- Updated Standing posture (Forward Crouch) ---
stand_hip_roll  =  0.00;
stand_hip_pitch =  -0.45;
stand_knee      =  0.90;
stand_ankle     = -0.45;

%% --- Updated Swing phase excursion ---
swing_hip_pitch_fwd =  -0.35;  
swing_knee_lift     =  0.60;
swing_ankle_up      =  -0.20;   
plant_hip_pitch     = -0.10;   

%% --- Pre-allocate joint arrays ---
% Each column: one joint, one leg
hip_roll_L  = zeros(N,1);
hip_pitch_L = zeros(N,1);
knee_L      = zeros(N,1);
ankle_L     = zeros(N,1);
hip_roll_R  = zeros(N,1);
hip_pitch_R = zeros(N,1);
knee_R      = zeros(N,1);
ankle_R     = zeros(N,1);

%% --- Fill joint trajectories ---
for k = 1:N
    tk = t(k);

    % Default: standing posture
    qL = [stand_hip_roll; stand_hip_pitch; stand_knee; stand_ankle];
    qR = [stand_hip_roll; stand_hip_pitch; stand_knee; stand_ankle];

    if tk >= t_start
        phase_t  = tk - t_start;
        step_idx = floor(phase_t / step_dur);
        t_in     = mod(phase_t, step_dur);
        swing    = step_dur * swing_frac;

        % Smooth blend using sine envelope
        sw = @(s) sin(pi * min(s,1));   % s = t_in/swing  in [0,1]

        if mod(step_idx, 2) == 0
            %% RIGHT foot swings, LEFT is support
            % LEFT (support)
            qL(2) = stand_hip_pitch + plant_hip_pitch * sw(t_in/swing);
            qL(3) = stand_knee;
            qL(4) = stand_ankle;

            % RIGHT (swing)
            if t_in < swing
                s = t_in / swing;
                qR(2) = stand_hip_pitch + swing_hip_pitch_fwd * sw(s);
                qR(3) = stand_knee      + swing_knee_lift      * sw(s);
                qR(4) = stand_ankle     + swing_ankle_up       * sw(s);
            else
                % landing phase: return to stance
                s = (t_in - swing) / (step_dur - swing);
                qR(2) = stand_hip_pitch + swing_hip_pitch_fwd*(1 - s);
                qR(3) = stand_knee;
                qR(4) = stand_ankle;
            end

        else
            %% LEFT foot swings, RIGHT is support
            % RIGHT (support)
            qR(2) = stand_hip_pitch + plant_hip_pitch * sw(t_in/swing);
            qR(3) = stand_knee;
            qR(4) = stand_ankle;

            % LEFT (swing)
            if t_in < swing
                s = t_in / swing;
                qL(2) = stand_hip_pitch + swing_hip_pitch_fwd * sw(s);
                qL(3) = stand_knee      + swing_knee_lift      * sw(s);
                qL(4) = stand_ankle     + swing_ankle_up       * sw(s);
            else
                s = (t_in - swing) / (step_dur - swing);
                qL(2) = stand_hip_pitch + swing_hip_pitch_fwd*(1 - s);
                qL(3) = stand_knee;
                qL(4) = stand_ankle;
            end
        end

        %% Hip roll: shift laterally over support foot
        roll_amp = 0.08;
        if mod(step_idx, 2) == 0
            % shift weight to left (support)
            qL(1) = -roll_amp * sw(t_in/step_dur);
            qR(1) =  roll_amp * sw(t_in/step_dur);
        else
            qL(1) =  roll_amp * sw(t_in/step_dur);
            qR(1) = -roll_amp * sw(t_in/step_dur);
        end
    end

    hip_roll_L(k)  = qL(1);
    hip_pitch_L(k) = qL(2);
    knee_L(k)      = qL(3);
    ankle_L(k)     = qL(4);
    hip_roll_R(k)  = qR(1);
    hip_pitch_R(k) = qR(2);
    knee_R(k)      = qR(3);
    ankle_R(k)     = qR(4);
end

%% Pack into timeseries
joint_data = [hip_roll_L, hip_pitch_L, knee_L, ankle_L, ...
              hip_roll_R, hip_pitch_R, knee_R, ankle_R];
joint_ref  = timeseries(joint_data, t);

fprintf('GaitScheduler: joint trajectories built – %d samples\n', N);
fprintf('  Joints: [hip_roll_L, hip_pitch_L, knee_L, ankle_L, | R-leg mirror]\n');

%% Quick plot for verification
figure('Name','Gait Joint Angles','Color','w');
subplot(2,2,1); plot(t, rad2deg([hip_pitch_L, hip_pitch_R]));
title('Hip Pitch'); ylabel('deg'); legend('L','R'); grid on;

subplot(2,2,2); plot(t, rad2deg([knee_L, knee_R]));
title('Knee'); ylabel('deg'); legend('L','R'); grid on;

subplot(2,2,3); plot(t, rad2deg([ankle_L, ankle_R]));
title('Ankle Pitch'); ylabel('deg'); legend('L','R'); grid on;

subplot(2,2,4); plot(t, rad2deg([hip_roll_L, hip_roll_R]));
title('Hip Roll'); ylabel('deg'); legend('L','R'); grid on;

sgtitle('Bipedal Walking – Joint Reference Trajectories');
