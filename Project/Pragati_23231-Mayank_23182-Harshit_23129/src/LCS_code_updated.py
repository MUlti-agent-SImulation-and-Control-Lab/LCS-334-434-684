import math
import time
import random
import os
from collections import deque, namedtuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

from controller import Robot, Camera, Lidar, GPS, Gyro, Keyboard, Supervisor
try:
    from vehicle import Driver
except Exception:
    Driver = None


TIME_STEP = 50  # ms

# action discretization (paper style reduced lattice-like)
ACT_V_COUNT = 4
ACT_W_COUNT = 9

DEFAULT_CRUISE = 20.0  # km/h baseline
MAX_CRUISE = 200.0

EPISODES = 1000
GAMMA = 0.95
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 3e-4
LEARNING_RATE_ALPHA = 3e-4
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 100000
MIN_REPLAY_SIZE = 512
TARGET_UPDATE_INTERVAL = 200
TRAIN_EVERY_N_STEPS = 1
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 30000  # relatively fast anneal

# reward / termination (matching paper)
MU_RDIS = 5.0
W_RDIS = 0.2
R_GOAL_POSE = 500.0
R_COLLISION = -200.0
R_GOAL_THRESHOLD = 1.0
HEADING_SCALE = 1.0
MU_ROBS = 3.0
COLLISION_DIST = 0.8
MAX_OBS_DIST = 5.0

# segment reward (milestone frequent to speed learning)
USE_SEGMENTS = True
SEGMENT_LENGTH = 2.5
SEGMENT_REWARD = 100.0

# termination caps
MAX_EPISODE_STEPS = 5000

# modalities settings
CAM_DOWNSAMPLE_FACTOR = 2
DEPTH_FRAME_H = 100   # target image crop height (paper used 100x100)
DEPTH_FRAME_W = 100

# stuck detection
STUCK_SPEED_TH = 0.2
STUCK_MAX_STEPS = 80

# misc
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ.setdefault('OMP_NUM_THREADS', '6')
os.environ.setdefault('MKL_NUM_THREADS', '6')

DEBUG = True
DEBUG_FREQ = 200

POSE_SAVE_DIR = "Results"
os.makedirs(POSE_SAVE_DIR, exist_ok=True)

CSV_LOG_PATH = os.path.join(POSE_SAVE_DIR, "episodes_log.csv")
STEP_CSV_PATH = os.path.join(POSE_SAVE_DIR, "steps_log.csv")
EPISODE_DIR = os.path.join(POSE_SAVE_DIR, "episode_steps")
os.makedirs(EPISODE_DIR, exist_ok=True)

# simple reward scaling to keep losses moderate
REWARD_SCALE = 10.0


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

def build_actions(act_v_count=ACT_V_COUNT, act_w_count=ACT_W_COUNT):
    actions = []
    for iv in range(act_v_count):
        v = float(iv) / max(1, (act_v_count - 1))
        for iw in range(act_w_count):
            w = -1.0 + float(iw) * (2.0 / max(1, (act_w_count - 1)))
            actions.append((v, w))
    return actions

ACTIONS = build_actions()


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayBuffer:
    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

shared_replay = ReplayBuffer()

# ----------------------
def safe_image_from_cam(camera, target_w=DEPTH_FRAME_W, target_h=DEPTH_FRAME_H):
    try:
        img = camera.getImage()
        if img is None:
            return None, False
        w = camera.getWidth()
        h = camera.getHeight()
        arr = np.frombuffer(img, dtype=np.uint8)
        expected = w*h*4
        if arr.size < expected:
            arr = np.resize(arr, expected)
        arr = arr.reshape((h, w, 4))
        rgb = (arr[:,:,2].astype(np.float32) + arr[:,:,1].astype(np.float32) + arr[:,:,0].astype(np.float32)) / (3.0*255.0)
        mid_row = h//2
        start_row = max(0, mid_row - target_h//2)
        crop = rgb[start_row:start_row+target_h, :]
        if crop.shape[1] != target_w:
            crop = np.array([np.interp(np.linspace(0, crop.shape[1]-1, target_w), np.arange(crop.shape[1]), crop[r,:]) for r in range(crop.shape[0])])
        return crop.astype(np.float32), True
    except Exception:
        return None, False

def lidar_to_point_features(lidar, num_bins=32, max_range=20.0):
    try:
        r = lidar.getRangeImage()
        if r is None:
            return np.ones(num_bins, dtype=np.float32)
        arr = np.array(r, dtype=np.float32)
        arr[arr <= 0.0] = max_range
        L = len(arr)
        bins = []
        step = max(1, L // num_bins)
        for i in range(num_bins):
            seg = arr[i*step:(i+1)*step]
            if seg.size == 0:
                val = max_range
            else:
                val = float(np.min(seg))
            bins.append(val)
        bins = np.array(bins, dtype=np.float32)
        bins = np.clip(bins, 0.0, max_range) / float(max_range)
        return bins
    except Exception:
        return np.ones(num_bins, dtype=np.float32)

# ----------------------

def try_init_torch(device_hint=None):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        from torch import amp
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.benchmark = True
        return torch, nn, F, optim, amp
    except Exception as e:
        if DEBUG:
            print("Torch init failed:", e)
        return None, None, None, None, None

def build_agent(action_count, device):
    torch, nn, F, optim, amp = try_init_torch()
    if torch is None:
        return None

    class PoseModule(nn.Module):
        def __init__(self, in_dim=4, h1=64, h2=128):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, h1)
            self.fc2 = nn.Linear(h1, h2)
        def forward(self, x):
            x = F.leaky_relu(self.fc1(x), 0.01)
            x = F.relu(self.fc2(x))
            return x

    class DepthModule(nn.Module):
        def __init__(self, in_c=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_c, 16, kernel_size=5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((6,6))
            self.fc = nn.Linear(6*6*64, 128)
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc(x))
            return x

    class PointModule(nn.Module):
        def __init__(self, in_dim=32, h=128):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, h)
            self.fc2 = nn.Linear(h, h)
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return x

    class FusionLSTM(nn.Module):
        def __init__(self, feat_dim, lstm_hidden=128, lstm_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True)
            self.out_fc = nn.Linear(lstm_hidden, lstm_hidden)
        def forward(self, x_seq):
            h, _ = self.lstm(x_seq)
            last = h[:, -1, :]
            out = F.relu(self.out_fc(last))
            return out

    class Actor(nn.Module):
        def __init__(self, obs_dim, action_dim=2, hidden=256, log_std_min=-20, log_std_max=2):
            super().__init__()
            self.log_std_min = log_std_min
            self.log_std_max = log_std_max
            self.fc1 = nn.Linear(obs_dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.mean = nn.Linear(hidden, action_dim)
            self.log_std = nn.Linear(hidden, action_dim)
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            mu = self.mean(x)
            log_std = self.log_std(x)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            return mu, log_std
        def sample(self, x):
            mu, log_std = self.forward(x)
            std = torch.exp(log_std)
            normal = torch.distributions.Normal(mu, std)
            z = normal.rsample()
            tanh_action = torch.tanh(z)
            log_prob = normal.log_prob(z) - torch.log(1 - tanh_action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            return tanh_action, log_prob, mu

    class QNetwork(nn.Module):
        def __init__(self, obs_dim, action_dim=2, hidden=256):
            super().__init__()
            self.fc1 = nn.Linear(obs_dim + action_dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.out = nn.Linear(hidden, 1)
        def forward(self, obs, action):
            x = torch.cat([obs, action], dim=1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.out(x)

    class Agent:
        def __init__(self, action_space, device):
            self.action_space = action_space
            self.action_count = len(action_space)
            self.device = device
            self.torch = torch
            self.amp = amp

            self.pose = PoseModule().to(device)
            self.depth = DepthModule().to(device)
            self.point = PointModule(in_dim=32).to(device)
            self.fuse_in_dim = 128 + 128 + 128
            self.lstm = FusionLSTM(self.fuse_in_dim, lstm_hidden=256).to(device)
            obs_dim = 256

            self.policy = Actor(obs_dim, action_dim=2).to(device)
            self.policy_target = Actor(obs_dim, action_dim=2).to(device)
            self.policy_target.load_state_dict(self.policy.state_dict())

            self.q1 = QNetwork(obs_dim, action_dim=2).to(device)
            self.q2 = QNetwork(obs_dim, action_dim=2).to(device)
            self.q1_target = QNetwork(obs_dim, action_dim=2).to(device)
            self.q2_target = QNetwork(obs_dim, action_dim=2).to(device)
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q2.state_dict())

            self.actor_optimizer = optim.Adam(list(self.policy.parameters()) + list(self.lstm.parameters()) + list(self.pose.parameters()) + list(self.depth.parameters()) + list(self.point.parameters()), lr=LEARNING_RATE_ACTOR, eps=1e-4, weight_decay=1e-6)
            self.critic_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=LEARNING_RATE_CRITIC, eps=1e-4, weight_decay=1e-6)

            self.target_entropy = -2.0
            self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float32, device=device, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LEARNING_RATE_ALPHA)

            self.alpha = self.log_alpha.exp().item()

            self.replay = ReplayBuffer()
            self.total_steps = 0
            self.train_steps = 0
            self.eps = EPS_START

            self.scaler = amp.GradScaler(enabled=(self.device.type == 'cuda'))

            self.tau = 0.005

        def act(self, state_dict, eval_mode=False):
            import torch as _torch
            try:
                seq_len = len(state_dict['pose_seq'])
                pose_feats = []
                depth_feats = []
                for i in range(seq_len):
                    p = _torch.tensor(state_dict['pose_seq'][i], dtype=_torch.float32, device=self.device).unsqueeze(0)
                    pf = self.pose(p)
                    pose_feats.append(pf)
                    d_np = state_dict['depth_seq'][i]
                    d_t = _torch.tensor(d_np, dtype=_torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                    df = self.depth(d_t)
                    depth_feats.append(df)
                point_np = np.asarray(state_dict['point'], dtype=np.float32)[None, :]
                pt_t = _torch.tensor(point_np, dtype=_torch.float32, device=self.device)
                pnf = self.point(pt_t)
                seq_feats = []
                for i in range(seq_len):
                    f = _torch.cat([pose_feats[i], depth_feats[i], pnf], dim=1)
                    seq_feats.append(f)
                seq_tensor = _torch.cat(seq_feats, dim=0).unsqueeze(0)
                with amp.autocast(device_type=self.device.type):
                    fused = self.lstm(seq_tensor)
                    if eval_mode:
                        mu, _ = self.policy.forward(fused)
                        raw_action = torch.tanh(mu).detach().cpu().numpy().flatten()
                    else:
                        tanh_action, logp, mu = self.policy.sample(fused)
                        raw_action = tanh_action.detach().cpu().numpy().flatten()
                v = float((raw_action[0] + 1.0) / 2.0)
                w = float(raw_action[1])
                v = clamp(v, 0.0, 1.0)
                w = clamp(w, -1.0, 1.0)
                acts = np.array(ACTIONS, dtype=np.float32)
                cont = np.array([v, w], dtype=np.float32)
                dists = np.sum((acts - cont) ** 2, axis=1)
                best_idx = int(np.argmin(dists))
                return v, w, best_idx
            except Exception as e:
                if DEBUG:
                    print("Act() exception (SAC):", e)
                idx = len(self.action_space)//2
                v, w = self.action_space[idx]
                return v, w, idx

        def push(self, *args):
            self.replay.push(*args)

        def train_step(self, batch_size=BATCH_SIZE):
            if len(self.replay) < max(MIN_REPLAY_SIZE, batch_size):
                return 0.0
            batch = self.replay.sample(batch_size)
            import torch as _torch

            seq_len = len(batch[0].state['pose_seq'])
            B = batch_size
            S = seq_len
            pose_arr = np.zeros((B, S, 4), dtype=np.float32)
            depth_arr = np.zeros((B, S, DEPTH_FRAME_H, DEPTH_FRAME_W), dtype=np.float32)
            point_arr = np.zeros((B, 32), dtype=np.float32)
            acts = np.zeros((B, 2), dtype=np.float32)
            rews = np.zeros((B,), dtype=np.float32)
            dones = np.zeros((B,), dtype=np.float32)
            next_pose_arr = np.zeros((B, S, 4), dtype=np.float32)
            next_depth_arr = np.zeros((B, S, DEPTH_FRAME_H, DEPTH_FRAME_W), dtype=np.float32)
            next_point_arr = np.zeros((B, 32), dtype=np.float32)

            for i, t in enumerate(batch):
                pose_arr[i, :, :] = np.stack(t.state['pose_seq'])
                depth_arr[i, :, :, :] = np.stack(t.state['depth_seq'])
                point_arr[i, :] = t.state['point']
                a = t.action
                if isinstance(a, (list, tuple, np.ndarray)):
                    acts[i, 0] = float(a[0])
                    acts[i, 1] = float(a[1])
                else:
                    try:
                        idx = int(a)
                        acts[i, 0] = ACTIONS[idx][0]
                        acts[i, 1] = ACTIONS[idx][1]
                    except Exception:
                        acts[i, 0] = ACTIONS[len(ACTIONS)//2][0]
                        acts[i, 1] = ACTIONS[len(ACTIONS)//2][1]
                rews[i] = float(t.reward)
                dones[i] = float(t.done)
                next_pose_arr[i, :, :] = np.stack(t.next_state['pose_seq'])
                next_depth_arr[i, :, :, :] = np.stack(t.next_state['depth_seq'])
                next_point_arr[i, :] = t.next_state['point']

            pose_b = _torch.from_numpy(pose_arr).to(self.device)
            depth_b = _torch.from_numpy(depth_arr).to(self.device)
            point_b = _torch.from_numpy(point_arr).to(self.device)
            acts_b = _torch.from_numpy(acts).to(self.device, dtype=_torch.float32)
            # simple reward scaling
            rews_t = _torch.from_numpy(rews).to(self.device, dtype=_torch.float32).unsqueeze(1) / REWARD_SCALE
            dones_t = _torch.from_numpy(dones).to(self.device, dtype=_torch.float32).unsqueeze(1)
            next_pose_b = _torch.from_numpy(next_pose_arr).to(self.device)
            next_depth_b = _torch.from_numpy(next_depth_arr).to(self.device)
            next_point_b = _torch.from_numpy(next_point_arr).to(self.device)

            pose_flat = pose_b.view(B*S, 4)
            depth_flat = depth_b.view(B*S, 1, DEPTH_FRAME_H, DEPTH_FRAME_W)
            point_flat = point_b.unsqueeze(1).repeat(1, S, 1).view(B*S, -1)

            device_type = self.device.type
            self.policy.train()
            self.q1.train()
            self.q2.train()
            with amp.autocast(device_type=device_type):
                pose_feat_flat = self.pose(pose_flat)
                depth_feat_flat = self.depth(depth_flat)
                point_feat_flat = self.point(point_flat)
                seq_feat = _torch.cat([pose_feat_flat, depth_feat_flat, point_feat_flat], dim=1)
                seq_feat = seq_feat.view(B, S, -1)
                fused = self.lstm(seq_feat)

                n_pose_flat = next_pose_b.view(B*S, 4)
                n_depth_flat = next_depth_b.view(B*S, 1, DEPTH_FRAME_H, DEPTH_FRAME_W)
                n_point_flat = next_point_b.unsqueeze(1).repeat(1, S, 1).view(B*S, -1)
                n_pose_feat_flat = self.pose(n_pose_flat)
                n_depth_feat_flat = self.depth(n_depth_flat)
                n_point_feat_flat = self.point(n_point_flat)
                n_seq_feat = _torch.cat([n_pose_feat_flat, n_depth_feat_flat, n_point_feat_flat], dim=1).view(B, S, -1)
                n_fused = self.lstm(n_seq_feat)

                with torch.no_grad():
                    n_tanh_a, n_logp, _ = self.policy.sample(n_fused)
                    n_a_cont = torch.zeros(B, 2, device=self.device)
                    n_a_cont[:, 0] = (n_tanh_a[:, 0] + 1.0) / 2.0
                    n_a_cont[:, 1] = n_tanh_a[:, 1]
                    q1_next = self.q1_target(n_fused, n_a_cont)
                    q2_next = self.q2_target(n_fused, n_a_cont)
                    q_next_min = torch.min(q1_next, q2_next)
                    alpha = torch.exp(self.log_alpha)
                    q_target = rews_t + (1.0 - dones_t) * GAMMA * (q_next_min - alpha * n_logp)

                q1_curr = self.q1(fused, acts_b)
                q2_curr = self.q2(fused, acts_b)
                critic_loss = F.mse_loss(q1_curr, q_target) + F.mse_loss(q2_curr, q_target)

            self.critic_optimizer.zero_grad()
            try:
                if self.device.type == 'cuda':
                    self.scaler.scale(critic_loss).backward()
                    try:
                        self.scaler.unscale_(self.critic_optimizer)
                    except Exception:
                        pass
                    torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 5.0)
                    try:
                        self.scaler.step(self.critic_optimizer)
                        self.scaler.update()
                    except Exception:
                        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 5.0)
                        self.critic_optimizer.step()
                else:
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 5.0)
                    self.critic_optimizer.step()
            except Exception as e:
                if DEBUG:
                    print("Critic update exception:", e)
                try:
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 5.0)
                    self.critic_optimizer.step()
                except Exception:
                    pass

            # recompute forward fresh for actor
            self.actor_optimizer.zero_grad()
            try:
                with amp.autocast(device_type=device_type):
                    pose_feat_flat2 = self.pose(pose_flat)
                    depth_feat_flat2 = self.depth(depth_flat)
                    point_feat_flat2 = self.point(point_flat)
                    seq_feat2 = _torch.cat([pose_feat_flat2, depth_feat_flat2, point_feat_flat2], dim=1)
                    seq_feat2 = seq_feat2.view(B, S, -1)
                    fused2 = self.lstm(seq_feat2)

                    tanh_a2, logp_a2, _ = self.policy.sample(fused2)
                    a_cont2 = torch.zeros(B, 2, device=self.device)
                    a_cont2[:, 0] = (tanh_a2[:, 0] + 1.0) / 2.0
                    a_cont2[:, 1] = tanh_a2[:, 1]
                    q1_pi2 = self.q1(fused2, a_cont2)
                    q2_pi2 = self.q2(fused2, a_cont2)
                    q_pi2 = torch.min(q1_pi2, q2_pi2)
                    alpha = torch.exp(self.log_alpha)
                    actor_loss = (alpha * logp_a2 - q_pi2).mean()

                if self.device.type == 'cuda':
                    self.scaler.scale(actor_loss).backward()
                    try:
                        self.scaler.unscale_(self.actor_optimizer)
                    except Exception:
                        pass
                    torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.lstm.parameters()) + list(self.pose.parameters()) + list(self.depth.parameters()) + list(self.point.parameters()), 5.0)
                    try:
                        self.scaler.step(self.actor_optimizer)
                        self.scaler.update()
                    except Exception:
                        torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.lstm.parameters()) + list(self.pose.parameters()) + list(self.depth.parameters()) + list(self.point.parameters()), 5.0)
                        self.actor_optimizer.step()
                else:
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.lstm.parameters()) + list(self.pose.parameters()) + list(self.depth.parameters()) + list(self.point.parameters()), 5.0)
                    self.actor_optimizer.step()
            except Exception as e:
                if DEBUG:
                    print("Actor update exception:", e)
                try:
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.lstm.parameters()) + list(self.pose.parameters()) + list(self.depth.parameters()) + list(self.point.parameters()), 5.0)
                    self.actor_optimizer.step()
                except Exception:
                    pass

            # alpha tuning
            try:
                self.alpha_optimizer.zero_grad()
                with amp.autocast(device_type=device_type):
                    pose_feat_flat3 = self.pose(pose_flat)
                    depth_feat_flat3 = self.depth(depth_flat)
                    point_feat_flat3 = self.point(point_flat)
                    seq_feat3 = _torch.cat([pose_feat_flat3, depth_feat_flat3, point_feat_flat3], dim=1)
                    seq_feat3 = seq_feat3.view(B, S, -1)
                    fused3 = self.lstm(seq_feat3)
                    _, logp_a3, _ = self.policy.sample(fused3)
                    alpha_loss = -(self.log_alpha * (logp_a3 + self.target_entropy).detach()).mean()

                if self.device.type == 'cuda':
                    self.scaler.scale(alpha_loss).backward()
                    try:
                        self.scaler.unscale_(self.alpha_optimizer)
                    except Exception:
                        pass
                    try:
                        self.scaler.step(self.alpha_optimizer)
                        self.scaler.update()
                    except Exception:
                        self.alpha_optimizer.step()
                else:
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                self.alpha = float(self.log_alpha.exp().detach().cpu().item())
            except Exception as e:
                if DEBUG:
                    print("Alpha update exception:", e)
                try:
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                except Exception:
                    pass

            try:
                for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.policy.parameters(), self.policy_target.parameters()):
                    target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)
            except Exception:
                pass

            self.train_steps += 1
            return float((critic_loss.detach().cpu().item() if isinstance(critic_loss, _torch.Tensor) else 0.0))

        def model_train_mode(self, mode=True):
            self.pose.train(mode)
            self.depth.train(mode)
            self.point.train(mode)
            self.lstm.train(mode)
            self.policy.train(mode)
            self.q1.train(mode)
            self.q2.train(mode)

        def get_trainable_params(self):
            return list(self.pose.parameters()) + list(self.depth.parameters()) + list(self.point.parameters()) + list(self.lstm.parameters()) + list(self.policy.parameters()) + list(self.q1.parameters()) + list(self.q2.parameters())

    import torch as _torch
    device = device
    agent = Agent(ACTIONS, device)
    return agent


# ----------------------
if 'robot' in globals() and isinstance(globals()['robot'], Robot):
    robot = globals()['robot']
else:
    robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = None
lidar = None
gps = None
gyro = None
keyboard = None
driver = None
supervisor = None

try:
    cam_dev = robot.getDevice("camera")
    if cam_dev:
        camera = cam_dev
        camera.enable(TIME_STEP)
        cam_w = camera.getWidth()
        cam_h = camera.getHeight()
except Exception:
    camera = None
    cam_w = cam_h = -1

try:
    ldev = robot.getDevice("Sick LMS 291")
    if ldev:
        lidar = ldev
        try:
            lidar.enable(TIME_STEP)
        except Exception:
            pass
except Exception:
    lidar = None

try:
    g = robot.getDevice("gps")
    if g:
        gps = g
        gps.enable(TIME_STEP)
except Exception:
    gps = None

try:
    gy = robot.getDevice("gyro")
    if gy:
        gyro = gy
        gyro.enable(TIME_STEP)
except Exception:
    gyro = None

keyboard = Keyboard()
keyboard.enable(TIME_STEP)

try:
    driver = Driver()
except Exception:
    driver = None

try:
    supervisor = Supervisor()
except Exception:
    supervisor = None

# logging & bookkeeping
episode_index = 0
episode_reward = 0.0
episodes_rewards = []
episodes_collisions = []
episode_step_counter = 0

# agent init placeholders (defer torch init a few steps)
agent = None
torch_enabled = False
deferred_init_attempted = False
DEFER_TORCH_STEPS = 4

SEQ_LEN = 4
pose_seq_buf = deque(maxlen=SEQ_LEN)
depth_seq_buf = deque(maxlen=SEQ_LEN)

shared_replay = ReplayBuffer()

# prev state and actions
prev_state = None
prev_action_idx = None  
prev_action_cont = None  
prev_dtarget = float('nan')

# other
heading = 0.0
dt = TIME_STEP / 1000.0

# diagnostics
step_counter = 0
pose_weight_norms = []  

if not os.path.exists(CSV_LOG_PATH):
    try:
        with open(CSV_LOG_PATH, mode='w', newline='') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["episode", "total_reward", "collision", "reason", "pose_weight_l2", "timestamp"])
    except Exception as e:
        if DEBUG:
            print("Could not create CSV log file:", e)

if not os.path.exists(STEP_CSV_PATH):
    try:
        with open(STEP_CSV_PATH, mode='w', newline='') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["global_step", "episode", "episode_step", "dobs", "yellow_px", "dev_norm", "reward_step", "action_idx", "collision_flag", "gps_x", "gps_y", "gps_z", "speed", "timestamp"])
    except Exception as e:
        if DEBUG:
            print("Could not create STEP CSV file:", e)

print("[controller] multimodal controller starting. EPISODES=", EPISODES)

# ----------------------

obstacle_time_steps = []
cumulative_deviation_time = []
rewards_over_time = []
collisions_over_time = []
global_step_idx = 0

episodes_obstacle_safe_pct = []
episodes_cumulative_deviation = []
episodes_rewards = []
episodes_collisions_counts = []

current_episode_step_count = 0
current_episode_safe_steps = 0
current_episode_abs_dev_sum = 0.0
current_episode_collision_count = 0

current_episode_steps_data = []

# ----------------------
def compute_reward(dtarget, prev_dtarget, dobs, angle_to_goal, yellow_px, yellow_valid, offroad_now):
    rdis = 0.0
    delta = 0.0
    try:
        if np.isfinite(prev_dtarget) and np.isfinite(dtarget):
            delta = int(prev_dtarget * (1.0/W_RDIS)) - int(dtarget * (1.0/W_RDIS))
            dsub = delta
            if dsub < 0:
                rdis = MU_RDIS * abs(dsub)
            else:
                rdis = -MU_RDIS * abs(dsub)
    except Exception:
        rdis = 0.0

    rcol = R_COLLISION if dobs < COLLISION_DIST else 0.0
    rhead = HEADING_SCALE * math.cos(angle_to_goal) if np.isfinite(angle_to_goal) else 0.0
    robs = 0.0
    if COLLISION_DIST <= dobs < MAX_OBS_DIST:
        robs = -math.exp(5.0 - dobs) / MU_ROBS

    r_yellow = 0.0
    if yellow_valid:
        px = abs(yellow_px)
        excess = max(0.0, px - 18)
        excess_norm = excess / (cam_w/2.0 if cam_w>0 else 1.0)
        r_yellow = -3.0 * excess_norm
        if px <= 18:
            r_yellow += 1.5

    total = rdis + rcol + rhead + robs + r_yellow
    return total, dict(rdis=rdis, rcol=rcol, rhead=rhead, robs=robs, r_yellow=r_yellow)


def save_the_8_plots():
    try:
        if len(obstacle_time_steps) > 0:
            xs = list(range(1, len(obstacle_time_steps)+1))
            plt.figure(figsize=(5, 4))
            plt.plot(xs, obstacle_time_steps, label='Min obstacle distance', linewidth=2, markersize=4)
            plt.xlabel('Time step', fontsize=10, fontweight='bold')
            plt.ylabel('Min obstacle distance (m)', fontsize=10, fontweight='bold')
            legend = plt.legend(fontsize=10, loc='best', prop={'weight': 'bold', 'size': 10})
            plt.grid(True, alpha=0.2)
            plt.xticks(fontsize=10, fontweight='bold')
            plt.yticks(fontsize=10, fontweight='bold')
            plt.tight_layout()
            fname = "01_obstacles_over_time.png"
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
            try:
                from shutil import copyfile
                copyfile(fname, os.path.join(POSE_SAVE_DIR, fname))
            except Exception:
                pass

        if len(episodes_obstacle_safe_pct) > 0:
            xs = list(range(1, len(episodes_obstacle_safe_pct)+1))
            plt.figure(figsize=(5, 4))
            plt.plot(xs, episodes_obstacle_safe_pct, label='Obstacle Safe %', linewidth=2, markersize=8)
            plt.xlabel('Episode', fontsize=10, fontweight='bold')
            plt.ylabel('Safe Steps (%)', fontsize=10, fontweight='bold')
            legend = plt.legend(fontsize=10, loc='best', prop={'weight': 'bold', 'size': 10})
            plt.grid(True, alpha=0.2)
            plt.xticks(fontsize=10, fontweight='bold')
            plt.yticks(fontsize=10, fontweight='bold')
            plt.tight_layout()
            fname = "02_obstacles_per_episode.png"
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
            try:
                from shutil import copyfile
                copyfile(fname, os.path.join(POSE_SAVE_DIR, fname))
            except Exception:
                pass

        if len(episodes_cumulative_deviation) > 0:
            xs = list(range(1, len(episodes_cumulative_deviation)+1))
            plt.figure(figsize=(5, 4))
            plt.plot(xs, episodes_cumulative_deviation, label='Cumulative deviation', linewidth=2, markersize=8)
            plt.xlabel('Episode', fontsize=10, fontweight='bold')
            plt.ylabel('Cumulative normalized deviation', fontsize=10, fontweight='bold')
            legend = plt.legend(fontsize=10, loc='best', prop={'weight': 'bold', 'size': 10})
            plt.grid(True, alpha=0.2)
            plt.xticks(fontsize=10, fontweight='bold')
            plt.yticks(fontsize=10, fontweight='bold')
            plt.tight_layout()
            fname = "03_cumulative_deviation_per_episode.png"
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
            try:
                from shutil import copyfile
                copyfile(fname, os.path.join(POSE_SAVE_DIR, fname))
            except Exception:
                pass

        if len(cumulative_deviation_time) > 0:
            xs = list(range(1, len(cumulative_deviation_time)+1))
            plt.figure(figsize=(5, 4))
            plt.plot(xs, cumulative_deviation_time, label='Cumulative deviation (time)', linewidth=2, markersize=4)
            plt.xlabel('Time step', fontsize=10, fontweight='bold')
            plt.ylabel('Cumulative normalized deviation', fontsize=10, fontweight='bold')
            legend = plt.legend(fontsize=10, loc='best', prop={'weight': 'bold', 'size': 10})
            plt.grid(True, alpha=0.2)
            plt.xticks(fontsize=10, fontweight='bold')
            plt.yticks(fontsize=10, fontweight='bold')
            plt.tight_layout()
            fname = "04_cumulative_deviation_over_time.png"
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
            try:
                from shutil import copyfile
                copyfile(fname, os.path.join(POSE_SAVE_DIR, fname))
            except Exception:
                pass

        if len(episodes_rewards) > 0:
            xs = list(range(1, len(episodes_rewards)+1))
            plt.figure(figsize=(5, 4))
            plt.plot(xs, episodes_rewards, label='Total reward', linewidth=2, markersize=8)
            plt.xlabel('Episode', fontsize=10, fontweight='bold')
            plt.ylabel('Total reward', fontsize=10, fontweight='bold')
            legend = plt.legend(fontsize=10, loc='best', prop={'weight': 'bold', 'size': 10})
            plt.grid(True, alpha=0.2)
            plt.xticks(fontsize=10, fontweight='bold')
            plt.yticks(fontsize=10, fontweight='bold')
            plt.tight_layout()
            fname = "05_total_reward_vs_episodes.png"
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
            try:
                from shutil import copyfile
                copyfile(fname, os.path.join(POSE_SAVE_DIR, fname))
            except Exception:
                pass

        if len(rewards_over_time) > 0:
            xs = list(range(1, len(rewards_over_time)+1))
            plt.figure(figsize=(5, 4))
            plt.plot(xs, rewards_over_time, label='Cumulative reward', linewidth=2, markersize=4)
            plt.xlabel('Time step', fontsize=10, fontweight='bold')
            plt.ylabel('Cumulative total reward', fontsize=10, fontweight='bold')
            legend = plt.legend(fontsize=10, loc='best', prop={'weight': 'bold', 'size': 10})
            plt.grid(True, alpha=0.2)
            plt.xticks(fontsize=10, fontweight='bold')
            plt.yticks(fontsize=10, fontweight='bold')
            plt.tight_layout()
            fname = "06_total_reward_vs_time.png"
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
            try:
                from shutil import copyfile
                copyfile(fname, os.path.join(POSE_SAVE_DIR, fname))
            except Exception:
                pass

        if len(collisions_over_time) > 0:
            xs = list(range(1, len(collisions_over_time)+1))
            plt.figure(figsize=(5, 4))
            plt.plot(xs, collisions_over_time, label='Cumulative collisions', linewidth=2, markersize=4)
            plt.xlabel('Time step', fontsize=10, fontweight='bold')
            plt.ylabel('Cumulative collisions', fontsize=10, fontweight='bold')
            legend = plt.legend(fontsize=10, loc='best', prop={'weight': 'bold', 'size': 10})
            plt.grid(True, alpha=0.2)
            plt.xticks(fontsize=10, fontweight='bold')
            plt.yticks(fontsize=10, fontweight='bold')
            plt.tight_layout()
            fname = "07_collision_count_vs_time.png"
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
            try:
                from shutil import copyfile
                copyfile(fname, os.path.join(POSE_SAVE_DIR, fname))
            except Exception:
                pass

        if len(episodes_collisions_counts) > 0:
            xs = list(range(1, len(episodes_collisions_counts)+1))
            plt.figure(figsize=(5, 4))
            plt.plot(xs, episodes_collisions_counts, label='Collisions per episode', linewidth=2, markersize=8)
            plt.xlabel('Episode', fontsize=10, fontweight='bold')
            plt.ylabel('Collision count', fontsize=10, fontweight='bold')
            legend = plt.legend(fontsize=10, loc='best', prop={'weight': 'bold', 'size': 10})
            plt.grid(True, alpha=0.2)
            plt.xticks(fontsize=10, fontweight='bold')
            plt.yticks(fontsize=10, fontweight='bold')
            plt.tight_layout()
            fname = "08_collision_counts_vs_episodes.png"
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
            try:
                from shutil import copyfile
                copyfile(fname, os.path.join(POSE_SAVE_DIR, fname))
            except Exception:
                pass
    except Exception as e:
        if DEBUG:
            print("Error saving 8 plots:", e)

def append_episode_csv(ep, total_reward, collision, reason, pose_l2):
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(CSV_LOG_PATH, mode='a', newline='') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([ep, f"{total_reward:.6f}", collision, reason if reason is not None else "", f"{pose_l2:.6f}", ts])
    except Exception as e:
        if DEBUG:
            print("Could not append to CSV:", e)

def append_step_csv(global_step, ep, ep_step, dobs, yellow_px, dev_norm, reward_step, action_idx, collision_flag, gps_coords, speed):
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(STEP_CSV_PATH, mode='a', newline='') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([global_step, ep, ep_step, f"{dobs:.6f}", f"{yellow_px:.6f}", f"{dev_norm:.6f}", f"{reward_step:.6f}", action_idx, collision_flag, f"{gps_coords[0]:.6f}" if gps_coords is not None else "", f"{gps_coords[1]:.6f}" if gps_coords is not None else "", f"{gps_coords[2]:.6f}" if gps_coords is not None else "", f"{speed:.6f}", ts])
    except Exception as e:
        if DEBUG:
            print("Could not append step CSV:", e)

# ----------------------
# MAIN LOOP
# ----------------------
try:
    while robot.step(TIME_STEP) != -1:
        if (not deferred_init_attempted) and (step_counter >= DEFER_TORCH_STEPS):
            deferred_init_attempted = True
            try:
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(device)
                agent = build_agent(len(ACTIONS), device)
                torch_enabled = (agent is not None)
                if torch_enabled:
                    transferred = 0
                    while len(shared_replay) > 0 and len(agent.replay) < agent.replay.capacity:
                        t = shared_replay.buffer.popleft()
                        agent.replay.push(t.state, t.action, t.reward, t.next_state, t.done)
                        transferred += 1
                    if DEBUG:
                        print(f"[replay-transfer] transferred {transferred} to agent.replay")
                else:
                    if DEBUG:
                        print("Agent build failed; continuing with stub behavior.")
            except Exception as e:
                print("Torch agent init exception:", e)
                agent = None
                torch_enabled = False

        k = keyboard.getKey()
        while k != -1:
            if k == Keyboard.UP:
                DEFAULT_CRUISE = clamp(DEFAULT_CRUISE + 2.0, 0.0, MAX_CRUISE)
            elif k == Keyboard.DOWN:
                DEFAULT_CRUISE = clamp(DEFAULT_CRUISE - 2.0, 0.0, MAX_CRUISE)
            k = keyboard.getKey()

        depth_frame = None
        depth_valid = False
        if camera is not None:
            depth_frame, depth_valid = safe_image_from_cam(camera, target_w=DEPTH_FRAME_W, target_h=DEPTH_FRAME_H)
            if depth_valid:
                pass

        point_feat = np.ones(32, dtype=np.float32)
        if lidar is not None:
            point_feat = lidar_to_point_features(lidar, num_bins=32, max_range=20.0)

        gps_coords = [math.nan, math.nan, math.nan]
        gps_speed = 0.0
        if gps is not None:
            try:
                vals = gps.getValues()
                gps_coords = [vals[0], vals[1], vals[2]]
                try:
                    gps_speed = gps.getSpeed()
                except Exception:
                    gps_speed = 0.0
            except Exception:
                pass

        if USE_SEGMENTS and gps is not None and not math.isnan(gps_coords[0]) and 'initial_x' not in globals():
            initial_x = gps_coords[0]
            last_milestone_x = initial_x
            goal_x = last_milestone_x + SEGMENT_LENGTH
            goal_z = gps_coords[2]
            if DEBUG:
                print(f"[segments] initial_x={initial_x:.2f}, first goal_x={goal_x:.2f}")

        dtarget = float('nan')
        angle_to_goal = 0.0
        if gps is not None and 'goal_x' in globals() and goal_x is not None:
            try:
                dx = goal_x - gps_coords[0]
                dz = goal_z - gps_coords[2]
                dtarget = math.hypot(dx, dz)
                angle_to_goal = math.atan2(dz, dx)
            except Exception:
                pass

        pose_w = 0.0
        if gyro is not None:
            try:
                gvals = gyro.getValues()
                if len(gvals) >= 3:
                    pose_w = gvals[2]
            except Exception:
                pass
        heading += pose_w * dt
        heading = (heading + math.pi) % (2*math.pi) - math.pi

        pose_theta = 0.0
        if not math.isnan(dtarget):
            pose_theta = angle_to_goal - heading
            pose_theta = (pose_theta + math.pi) % (2*math.pi) - math.pi

        pose_vec = np.array([dtarget if np.isfinite(dtarget) else 9999.0, pose_theta, gps_speed, pose_w], dtype=np.float32)
        pose_seq_buf.append(pose_vec)
        if depth_valid:
            depth_seq_buf.append(depth_frame)
        else:
            depth_seq_buf.append(np.zeros((DEPTH_FRAME_H, DEPTH_FRAME_W), dtype=np.float32))

        dobs = MAX_OBS_DIST
        if lidar is not None:
            try:
                r = np.array(lidar.getRangeImage(), dtype=np.float32)
                r[r<=0.01] = MAX_OBS_DIST
                dobs = float(np.min(r))
            except Exception:
                dobs = MAX_OBS_DIST

        yellow_px = 0.0
        yellow_valid = False
        if camera is not None and depth_valid:
            try:
                mid = depth_frame.shape[0]//2
                row = depth_frame[mid]
                xs = np.arange(row.shape[0])
                cx = np.sum(xs * row) / (np.sum(row) + 1e-6)
                center_x = (row.shape[0]-1)/2.0
                yellow_px = cx - center_x
                yellow_valid = True
            except Exception:
                yellow_valid = False

        global_step_idx += 1
        try:
            current_episode_step_count += 1
        except Exception:
            current_episode_step_count = 1

        try:
            if dobs > COLLISION_DIST:
                current_episode_safe_steps += 1
        except Exception:
            pass

        try:
            obstacle_time_steps.append(min(dobs, MAX_OBS_DIST))
        except Exception:
            obstacle_time_steps.append(MAX_OBS_DIST)

        try:
            if yellow_valid and cam_w > 0:
                dev = abs(yellow_px) / (cam_w/2.0)
            else:
                dev = 1.0
            dev = float(max(0.0, min(dev, 10.0)))
        except Exception:
            dev = 1.0

        try:
            last_cum = cumulative_deviation_time[-1] if len(cumulative_deviation_time)>0 else 0.0
            cumulative_deviation_time.append(last_cum + dev)
        except Exception:
            cumulative_deviation_time.append(dev)

        done = False
        reason = None
        if dobs < COLLISION_DIST:
            done = True
            reason = "collision"

        if gps_speed < STUCK_SPEED_TH:
            if 'stuck_counter' not in globals():
                stuck_counter = 0
                globals()['stuck_counter'] = 0
            globals()['stuck_counter'] += 1
            if globals()['stuck_counter'] >= STUCK_MAX_STEPS:
                done = True
                reason = "stuck"
        else:
            if 'stuck_counter' in globals():
                globals()['stuck_counter'] = 0

        total_r = 0.0
        comps = {}
        if prev_state is not None and np.isfinite(dtarget):
            total_r, comps = compute_reward(dtarget, prev_dtarget, dobs, pose_theta, yellow_px, yellow_valid, False)
            next_state = {
                'pose_seq': list(pose_seq_buf) if len(pose_seq_buf)==SEQ_LEN else list(pose_seq_buf)+[pose_vec]*(SEQ_LEN-len(pose_seq_buf)),
                'depth_seq': list(depth_seq_buf) if len(depth_seq_buf)==SEQ_LEN else list(depth_seq_buf)+[np.zeros((DEPTH_FRAME_H,DEPTH_FRAME_W),dtype=np.float32)]*(SEQ_LEN-len(depth_seq_buf)),
                'point': point_feat.copy()
            }
            r_to_store = float(max(-1000.0, min(1000.0, total_r)))
            # store continuous action in replay for SAC training
            try:
                if agent is not None:
                    act_to_store = prev_action_cont if prev_action_cont is not None else prev_action_idx
                    agent.push(prev_state, act_to_store, r_to_store/REWARD_SCALE, next_state, done)
                else:
                    act_to_store = prev_action_cont if prev_action_cont is not None else prev_action_idx
                    shared_replay.push(prev_state, act_to_store, r_to_store/REWARD_SCALE, next_state, done)
            except Exception:
                try:
                    shared_replay.push(prev_state, prev_action_idx, r_to_store/REWARD_SCALE, next_state, done)
                except Exception:
                    pass
            episode_reward += total_r

        if USE_SEGMENTS and gps is not None and 'last_milestone_x' in globals() and not math.isnan(gps_coords[0]):
            try:
                while gps_coords[0] - last_milestone_x >= SEGMENT_LENGTH:
                    center_factor = 1.0
                    if yellow_valid:
                        if abs(yellow_px) > 60:
                            center_factor = 0.5
                    awarded = SEGMENT_REWARD * center_factor
                    episode_reward += awarded
                    total_r += awarded
                    last_milestone_x += SEGMENT_LENGTH
                    goal_x = last_milestone_x + SEGMENT_LENGTH
                    if DEBUG:
                        print(f"[segment] passed milestone -> awarded {awarded:.1f}")
            except Exception:
                pass

        if 'episode_step' not in globals():
            globals()['episode_step'] = 0
        globals()['episode_step'] += 1
        if globals()['episode_step'] >= MAX_EPISODE_STEPS:
            done = True
            reason = "max_steps"

        state_for_agent = {
            'pose_seq': list(pose_seq_buf) if len(pose_seq_buf)==SEQ_LEN else [pose_vec]*(SEQ_LEN-len(pose_seq_buf)) + list(pose_seq_buf),
            'depth_seq': list(depth_seq_buf) if len(depth_seq_buf)==SEQ_LEN else [np.zeros((DEPTH_FRAME_H,DEPTH_FRAME_W),dtype=np.float32)]*(SEQ_LEN-len(depth_seq_buf)) + list(depth_seq_buf),
            'point': point_feat.copy()
        }

        if agent is not None:
            try:
                v, w, act_idx = agent.act(state_for_agent, eval_mode=False)
            except Exception as e:
                if DEBUG:
                    print("agent.act error:", e)
                act_idx = len(ACTIONS)//2
                v, w = ACTIONS[act_idx]
        else:
            act_idx = len(ACTIONS)//2
            v, w = ACTIONS[act_idx]

        # store both index for logging and continuous action for replay
        prev_action_idx = act_idx
        prev_action_cont = (float(v), float(w))

        try:
            if agent is not None:
                agent.total_steps += 1
        except Exception:
            pass

        target_kmh = DEFAULT_CRUISE + v * 40.0
        try:
            if driver is not None:
                driver.setCruisingSpeed(target_kmh)
                driver.setSteeringAngle(w * 0.3)
                cur_sp = driver.getCurrentSpeed()
                err = target_kmh - cur_sp
                thr = max(0.0, min(1.0, 0.01 * err))
                if thr > 0.02:
                    driver.setThrottle(thr)
                else:
                    driver.setThrottle(0.0)
            else:
                cur_sp = 0.0
        except Exception:
            cur_sp = 0.0

        collision_flag = 1 if dobs < COLLISION_DIST else 0
        if collision_flag:
            current_episode_collision_count += 1

        try:
            append_step_csv(global_step_idx, episode_index+1, current_episode_step_count, dobs, yellow_px, dev, total_r, act_idx, collision_flag, gps_coords, gps_speed)
        except Exception:
            pass

        try:
            current_episode_steps_data.append({
                "global_step": global_step_idx,
                "episode": episode_index+1,
                "episode_step": current_episode_step_count,
                "dobs": dobs,
                "yellow_px": yellow_px,
                "dev_norm": dev,
                "reward_step": total_r,
                "action_idx": act_idx,
                "collision_flag": collision_flag,
                "gps_x": gps_coords[0],
                "gps_y": gps_coords[1],
                "gps_z": gps_coords[2],
                "speed": gps_speed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            })
        except Exception:
            pass

        try:
            last_reward_cum = rewards_over_time[-1] if len(rewards_over_time)>0 else 0.0
            rewards_over_time.append(last_reward_cum + total_r)
        except Exception:
            rewards_over_time.append(total_r)

        try:
            last_coll_cum = collisions_over_time[-1] if len(collisions_over_time)>0 else 0
            collisions_over_time.append(last_coll_cum + collision_flag)
        except Exception:
            collisions_over_time.append(collision_flag)

        prev_state = {
            'pose_seq': state_for_agent['pose_seq'],
            'depth_seq': state_for_agent['depth_seq'],
            'point': state_for_agent['point']
        }
        prev_dtarget = dtarget

        try:
            if agent is not None and len(agent.replay) >= MIN_REPLAY_SIZE and (agent.total_steps % TRAIN_EVERY_N_STEPS == 0):
                loss = agent.train_step(BATCH_SIZE)
                if DEBUG and (step_counter % DEBUG_FREQ == 0):
                    print(f"[train] total_steps={agent.total_steps}, loss={loss:.6f}, replay_len={len(agent.replay)}")
        except Exception as e:
            if DEBUG and (step_counter % DEBUG_FREQ == 0):
                print("Train error:", e)

        try:
            if (step_counter % DEBUG_FREQ == 0) or (globals().get('episode_step', 0) == 0):
                save_the_8_plots()
        except Exception as e:
            if DEBUG:
                print("Periodic plot save error:", e)

        if DEBUG and (step_counter % DEBUG_FREQ == 0):
            print(f"[debug] step={step_counter}, ep={episode_index+1}, action_idx={act_idx}, dobs={dobs:.2f}, dtarget={dtarget:.2f}, R_step={total_r:.3f}")

        if done:
            ep_num = episode_index + 1
            episodes_rewards.append(episode_reward)
            episodes_collisions.append(1 if reason in ("collision",) else 0)

            try:
                if agent is not None:
                    import torch as _torch
                    total_norm_sq = 0.0
                    for p in agent.pose.parameters():
                        arr = p.detach().cpu().numpy()
                        total_norm_sq += float((arr**2).sum())
                    l2 = math.sqrt(total_norm_sq) if total_norm_sq>0.0 else 0.0
                    pose_weight_norms.append(l2)
                else:
                    l2 = 0.0
                    pose_weight_norms.append(0.0)
            except Exception:
                l2 = 0.0
                pose_weight_norms.append(0.0)

            try:
                safe_pct = 100.0 * (current_episode_safe_steps / max(1, current_episode_step_count))
            except Exception:
                safe_pct = 0.0
            try:
                episodes_obstacle_safe_pct.append(safe_pct)
            except Exception:
                pass
            try:
                episodes_cumulative_deviation.append(float(current_episode_abs_dev_sum))
            except Exception:
                episodes_cumulative_deviation.append(0.0)
            try:
                episodes_collisions_counts.append(int(current_episode_collision_count))
            except Exception:
                episodes_collisions_counts.append(0)

            current_episode_step_count = 0
            current_episode_safe_steps = 0
            current_episode_abs_dev_sum = 0.0
            current_episode_collision_count = 0

            try:
                ep_csv_path = os.path.join(EPISODE_DIR, f"episode_{ep_num:04d}_steps.csv")
                with open(ep_csv_path, mode='w', newline='') as fcsv:
                    fieldnames = ["global_step","episode","episode_step","dobs","yellow_px","dev_norm","reward_step","action_idx","collision_flag","gps_x","gps_y","gps_z","speed","timestamp"]
                    writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in current_episode_steps_data:
                        writer.writerow(row)
                current_episode_steps_data = []
            except Exception as e:
                if DEBUG:
                    print("Could not save individual episode CSV:", e)

            try:
                print(f"Episode {ep_num} ended -> total_reward={episode_reward:.2f}, collision={episodes_collisions[-1]}, reason={reason}")
            except Exception:
                print(f"Episode {ep_num} ended -> total_reward={episode_reward:.2f}")

            try:
                append_episode_csv(ep_num, episode_reward, episodes_collisions[-1], reason, l2)
            except Exception as e:
                if DEBUG:
                    print("CSV append error:", e)

            try:
                save_the_8_plots()
            except Exception as e:
                if DEBUG:
                    print("Error saving plots:", e)

            episode_index += 1
            episode_reward = 0.0
            globals()['episode_step'] = 0
            prev_state = None
            prev_action_idx = None
            prev_action_cont = None
            prev_dtarget = float('nan')
            pose_seq_buf.clear()
            depth_seq_buf.clear()
            if 'stuck_counter' in globals():
                globals()['stuck_counter'] = 0
            try:
                if supervisor is not None:
                    supervisor.simulationReset()
                    for _ in range(6):
                        robot.step(TIME_STEP)
            except Exception:
                pass

            if agent is not None:
                try:
                    transferred = 0
                    while len(shared_replay) > 0 and len(agent.replay) < agent.replay.capacity:
                        t = shared_replay.buffer.popleft()
                        agent.replay.push(t.state, t.action, t.reward, t.next_state, t.done)
                        transferred += 1
                    if transferred and DEBUG:
                        print(f"[replay-fill] transferred {transferred} to agent.replay (post-episode)")
                except Exception:
                    pass

            if EPISODES is not None and episode_index >= EPISODES:
                print("Completed requested episodes; exiting controller loop.")
                break

        step_counter += 1

except KeyboardInterrupt:
    print("Interrupted by user.")
except Exception as e:
    print("Controller exception:", e)

try:
    if len(episodes_rewards) > 0:
        save_the_8_plots()
        if DEBUG:
            print("Saved final 8 plots.")
except Exception as e:
    print("Warning saving plots:", e)

try:
    import torch
    if agent is not None:
        try:
            torch_save_base = os.path.join(POSE_SAVE_DIR, "model_weights")
            os.makedirs(torch_save_base, exist_ok=True)
            torch.save(agent.pose.state_dict(), os.path.join(torch_save_base, "pose_module.pth"))
            torch.save(agent.depth.state_dict(), os.path.join(torch_save_base, "depth_module.pth"))
            torch.save(agent.point.state_dict(), os.path.join(torch_save_base, "point_module.pth"))
            torch.save(agent.lstm.state_dict(), os.path.join(torch_save_base, "lstm_module.pth"))
            torch.save(agent.policy.state_dict(), os.path.join(torch_save_base, "policy_net.pth"))
            try:
                torch.save(agent.policy_target.state_dict(), os.path.join(torch_save_base, "policy_target_net.pth"))
            except Exception:
                pass
            try:
                torch.save(agent.q1.state_dict(), os.path.join(torch_save_base, "q1_net.pth"))
                torch.save(agent.q2.state_dict(), os.path.join(torch_save_base, "q2_net.pth"))
                torch.save(agent.q1_target.state_dict(), os.path.join(torch_save_base, "q1_target_net.pth"))
                torch.save(agent.q2_target.state_dict(), os.path.join(torch_save_base, "q2_target_net.pth"))
            except Exception:
                pass
            try:
                torch.save(agent.actor_optimizer.state_dict(), os.path.join(torch_save_base, "actor_optimizer.pth"))
            except Exception:
                pass
            try:
                torch.save(agent.critic_optimizer.state_dict(), os.path.join(torch_save_base, "critic_optimizer.pth"))
            except Exception:
                pass
            if DEBUG:
                print(f"Model weights saved to {torch_save_base}")
        except Exception as e:
            print("Could not save model weights:", e)
except Exception:
    pass

print("Controller exiting.")
