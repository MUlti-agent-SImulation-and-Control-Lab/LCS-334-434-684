%% ref.m  –  ZMP staircase reference for bipedal walking
%  Generates a 4-output reference: [CoM_x_ref; ZMP_x_ref; CoM_y_ref; ZMP_y_ref]

Ts        = 0.05;
t         = (0:Ts:10)';    % 10-second walk
N         = length(t);

step_dur  = 0.5;           % seconds per step (one foot)
step_len  = 0.60;          % forward stride length (m)
foot_sep  = 0.18;          % lateral distance between feet centre-lines (m)

% ---- pre-allocate ----
zmp_x = zeros(N,1);
zmp_y = zeros(N,1);

% ---- build ZMP staircase ----
%   Phase 0 : double support standing
%   Then alternating steps: R foot -> L foot -> R foot -> ...

t_start = 0.5;   % start walking after 0.5 s
foot    = 0;     % 0 = right support, 1 = left support
x_foot  = 0;     % current support foot x position

for k = 1:N
    tk = t(k);
    if tk < t_start
        % standing still – ZMP at origin
        zmp_x(k) = 0;
        zmp_y(k) = 0;
    else
        phase_t  = tk - t_start;
        step_idx = floor(phase_t / step_dur);   % which step number
        t_in     = mod(phase_t, step_dur);       % time within step

        % Support foot alternates each step
        cur_foot = mod(step_idx, 2);             % 0=right, 1=left
        x_foot   = step_idx * step_len / 2;     % feet advance each step

        % lateral: right foot = -foot_sep/2, left foot = +foot_sep/2
        y_foot   = (cur_foot == 0) * (-foot_sep/2) + ...
                   (cur_foot == 1) * ( foot_sep/2);

        % smooth ZMP shift using half-cosine blend (avoids sharp steps)
        blend = 0.5*(1 - cos(pi * t_in / step_dur));
        if step_idx == 0
            % first step: shift from centre
            zmp_x(k) = blend * x_foot;
            zmp_y(k) = blend * y_foot;
        else
            prev_foot = mod(step_idx-1, 2);
            x_prev    = (step_idx-1) * step_len / 2;
            y_prev    = (prev_foot==0)*(-foot_sep/2) + (prev_foot==1)*(foot_sep/2);
            zmp_x(k)  = x_prev + blend*(x_foot - x_prev);
            zmp_y(k)  = y_prev + blend*(y_foot - y_prev);
        end
    end
end

% ---- CoM reference: slightly ahead of ZMP (leaning into step) ----
% Simple: CoM_x tracks a smooth ramp at walking speed
walk_spd  = step_len / step_dur;     % ~0.25 m/s
com_x_ref = max(0, t - t_start) * walk_spd;
com_y_ref = zeros(N,1);              % keep lateral CoM centred

% ---- pack into timeseries ----
%   columns: [CoM_x_ref, ZMP_x_ref, CoM_y_ref, ZMP_y_ref]
ref_data = [com_x_ref, zmp_x, com_y_ref, zmp_y];
ref_ts   = timeseries(ref_data, t);

fprintf('ref.m: walking reference built – %d samples, %.1f s\n', N, t(end));
fprintf('  Walk speed : %.3f m/s\n', walk_spd);
fprintf('  Step length: %.3f m\n', step_len);
fprintf('  Step period: %.3f s\n', step_dur);
