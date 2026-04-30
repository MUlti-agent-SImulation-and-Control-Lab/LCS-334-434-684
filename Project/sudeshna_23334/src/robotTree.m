%% robotTree.m - Full 2D Bipedal Robot
% Each leg: Hip Roll + Hip Pitch + Knee Pitch + Ankle Pitch
% Total movable joints: 8 (4 per leg)

robot = rigidBodyTree('DataFormat','column','MaxNumBodies',20);

%% ---- PHYSICAL PARAMETERS (tune to your robot) ----
thigh_len  = 0.40;   % m  (hip   -> knee)
shank_len  = 0.40;   % m  (knee  -> ankle)
foot_len   = 0.20;   % m  (ankle -> toe)
hip_width  = 0.10;   % m  (lateral offset from base to each hip)

%% --- LEFT LEG ---

% ---- Hip Roll (lateral abduction/adduction) ----
body_hip_roll_L            = rigidBody('hip_roll_L');
joint_hip_roll_L           = rigidBodyJoint('hip_roll_L_jnt', 'revolute');
joint_hip_roll_L.JointAxis = [1 0 0];
setFixedTransform(joint_hip_roll_L, trvec2tform([0, -hip_width, 0]));
body_hip_roll_L.Joint      = joint_hip_roll_L;
addBody(robot, body_hip_roll_L, 'base');

% ---- Hip Pitch (sagittal flexion/extension) ----
body_hip_pitch_L            = rigidBody('hip_pitch_L');
joint_hip_pitch_L           = rigidBodyJoint('hip_pitch_L_jnt', 'revolute');
joint_hip_pitch_L.JointAxis = [0 1 0];
setFixedTransform(joint_hip_pitch_L, trvec2tform([0, 0, 0]));
body_hip_pitch_L.Joint      = joint_hip_pitch_L;
addBody(robot, body_hip_pitch_L, 'hip_roll_L');

% ---- Thigh (rigid link) ----
body_thigh_L           = rigidBody('thigh_L');
joint_thigh_L          = rigidBodyJoint('thigh_L_jnt', 'fixed');
setFixedTransform(joint_thigh_L, trvec2tform([0, 0, -thigh_len]));
body_thigh_L.Joint     = joint_thigh_L;
addBody(robot, body_thigh_L, 'hip_pitch_L');

% ---- Knee Pitch ----
body_knee_L            = rigidBody('knee_L');
joint_knee_L           = rigidBodyJoint('knee_L_jnt', 'revolute');
joint_knee_L.JointAxis = [0 1 0];
setFixedTransform(joint_knee_L, trvec2tform([0, 0, 0]));
body_knee_L.Joint      = joint_knee_L;
addBody(robot, body_knee_L, 'thigh_L');

% ---- Shank (rigid link) ----
body_shank_L           = rigidBody('shank_L');
joint_shank_L          = rigidBodyJoint('shank_L_jnt', 'fixed');
setFixedTransform(joint_shank_L, trvec2tform([0, 0, -shank_len]));
body_shank_L.Joint     = joint_shank_L;
addBody(robot, body_shank_L, 'knee_L');

% ---- Ankle Pitch ----
body_ankle_L            = rigidBody('ankle_L');
joint_ankle_L           = rigidBodyJoint('ankle_L_jnt', 'revolute');
joint_ankle_L.JointAxis = [0 1 0];
setFixedTransform(joint_ankle_L, trvec2tform([0, 0, 0]));
body_ankle_L.Joint      = joint_ankle_L;
addBody(robot, body_ankle_L, 'shank_L');

% ---- Foot (rigid link) ----
body_foot_L            = rigidBody('foot_L');
joint_foot_L           = rigidBodyJoint('foot_L_jnt', 'fixed');
setFixedTransform(joint_foot_L, trvec2tform([foot_len/2, 0, -0.05]));
body_foot_L.Joint      = joint_foot_L;
addBody(robot, body_foot_L, 'ankle_L');

%% --- RIGHT LEG ---

% ---- Hip Roll (lateral abduction/adduction) ----
body_hip_roll_R            = rigidBody('hip_roll_R');
joint_hip_roll_R           = rigidBodyJoint('hip_roll_R_jnt', 'revolute');
joint_hip_roll_R.JointAxis = [1 0 0];
setFixedTransform(joint_hip_roll_R, trvec2tform([0, hip_width, 0]));
body_hip_roll_R.Joint      = joint_hip_roll_R;
addBody(robot, body_hip_roll_R, 'base');

% ---- Hip Pitch (sagittal flexion/extension) ----
body_hip_pitch_R            = rigidBody('hip_pitch_R');
joint_hip_pitch_R           = rigidBodyJoint('hip_pitch_R_jnt', 'revolute');
joint_hip_pitch_R.JointAxis = [0 1 0];
setFixedTransform(joint_hip_pitch_R, trvec2tform([0, 0, 0]));
body_hip_pitch_R.Joint      = joint_hip_pitch_R;
addBody(robot, body_hip_pitch_R, 'hip_roll_R');

% ---- Thigh (rigid link) ----
body_thigh_R           = rigidBody('thigh_R');
joint_thigh_R          = rigidBodyJoint('thigh_R_jnt', 'fixed');
setFixedTransform(joint_thigh_R, trvec2tform([0, 0, -thigh_len]));
body_thigh_R.Joint     = joint_thigh_R;
addBody(robot, body_thigh_R, 'hip_pitch_R');

% ---- Knee Pitch ----
body_knee_R            = rigidBody('knee_R');
joint_knee_R           = rigidBodyJoint('knee_R_jnt', 'revolute');
joint_knee_R.JointAxis = [0 1 0];
setFixedTransform(joint_knee_R, trvec2tform([0, 0, 0]));
body_knee_R.Joint      = joint_knee_R;
addBody(robot, body_knee_R, 'thigh_R');

% ---- Shank (rigid link) ----
body_shank_R           = rigidBody('shank_R');
joint_shank_R          = rigidBodyJoint('shank_R_jnt', 'fixed');
setFixedTransform(joint_shank_R, trvec2tform([0, 0, -shank_len]));
body_shank_R.Joint     = joint_shank_R;
addBody(robot, body_shank_R, 'knee_R');

% ---- Ankle Pitch ----
body_ankle_R            = rigidBody('ankle_R');
joint_ankle_R           = rigidBodyJoint('ankle_R_jnt', 'revolute');
joint_ankle_R.JointAxis = [0 1 0];
setFixedTransform(joint_ankle_R, trvec2tform([0, 0, 0]));
body_ankle_R.Joint      = joint_ankle_R;
addBody(robot, body_ankle_R, 'shank_R');

% ---- Foot (rigid link) ----
body_foot_R            = rigidBody('foot_R');
joint_foot_R           = rigidBodyJoint('foot_R_jnt', 'fixed');
setFixedTransform(joint_foot_R, trvec2tform([foot_len/2, 0, -0.05]));
body_foot_R.Joint      = joint_foot_R;
addBody(robot, body_foot_R, 'ankle_R');

%% --- JOINT LIMITS ---

% Helper: find body index by name matching — no toolbox function needed
getIdx = @(name) find(cellfun(@(b) strcmp(b.Name, name), robot.Bodies));

% Hip Roll : ±20 deg
robot.Bodies{getIdx('hip_roll_L')}.Joint.PositionLimits  = deg2rad([-20,  20]);
robot.Bodies{getIdx('hip_roll_R')}.Joint.PositionLimits  = deg2rad([-20,  20]);

% Hip Pitch : -30 (extension) to +90 (flexion)
robot.Bodies{getIdx('hip_pitch_L')}.Joint.PositionLimits = deg2rad([-30,  90]);
robot.Bodies{getIdx('hip_pitch_R')}.Joint.PositionLimits = deg2rad([-30,  90]);

% Knee Pitch : 0 to 120 deg (no hyperextension)
robot.Bodies{getIdx('knee_L')}.Joint.PositionLimits      = deg2rad([  0, 120]);
robot.Bodies{getIdx('knee_R')}.Joint.PositionLimits      = deg2rad([  0, 120]);

% Ankle Pitch : ±30 deg
robot.Bodies{getIdx('ankle_L')}.Joint.PositionLimits     = deg2rad([-30,  30]);
robot.Bodies{getIdx('ankle_R')}.Joint.PositionLimits     = deg2rad([-30,  30]);

%% --- VERIFY ---

% 1. Print all bodies and parents
fprintf('\n======= Robot Body Tree =======\n');
fprintf('Total bodies : %d\n', robot.NumBodies);
fprintf('%-5s %-20s %-20s %-15s\n','Idx','Body','Parent','Joint Type');
fprintf('%s\n', repmat('-',1,65));
for i = 1:robot.NumBodies
    b = robot.Bodies{i};
    fprintf('%-5d %-20s %-20s %-15s\n', ...
        i, b.Name, b.Parent.Name, b.Joint.Type);
end

% 2. Print joint limits for revolute joints only
fprintf('\n======= Joint Limits =======\n');
fprintf('%-20s  %8s  %8s\n','Body','Min(deg)','Max(deg)');
fprintf('%s\n', repmat('-',1,42));
for i = 1:robot.NumBodies
    b = robot.Bodies{i};
    if strcmp(b.Joint.Type, 'revolute')
        lims = rad2deg(b.Joint.PositionLimits);
        fprintf('%-20s  %8.1f  %8.1f\n', b.Name, lims(1), lims(2));
    end
end

% 3. Visual check
showdetails(robot);
