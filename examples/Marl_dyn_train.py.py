"""
Training script for dynamic waypoint multi-agent environment (centralized PPO).

- Train with VecMonitor + VecNormalize (train & eval wrapped identically)
- Print live metrics to stdout and TensorBoard
- Evaluate periodically DURING training (no separate retrain)

Usage (typical):
  python Marl_dyn_train.py.py --timesteps 500000 --num-drones 4 --num-waypoints 20 \
    --waypoint-radius 0.5 --waypoint-hold-steps 5 --ctrl-freq 60

Quick test:
  python Marl_dyn_train.py.py --quick
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import random
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import torch.nn as nn
# --------------------------
# Robust import of the env
# --------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(CURRENT_DIR)
PARENT_ROOT = os.path.dirname(PKG_ROOT)
for path in [PARENT_ROOT, PKG_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

MultiAgentReinforcementLearning = None
try:
    # Try package-style path (common layout)
    from gym_pybullet_drones.envs.Marl_dynamic_waypoints import MultiAgentReinforcementLearning  # type: ignore
except Exception:
    try:
        from envs.Marl_dynamic_waypoints import MultiAgentReinforcementLearning  # type: ignore
    except Exception:
        # Fallback: load directly by filename next to this script
        import importlib.util
        candidate = os.path.join(PKG_ROOT, 'env', 'Marl_dynamic_waypoints.py')
        alt_candidate = os.path.join(CURRENT_DIR, 'Marl_dynamic_waypoints.py')
        for c in [candidate, alt_candidate]:
            if os.path.exists(c):
                spec = importlib.util.spec_from_file_location('marl_dynamic_waypoints', c)
                mod = importlib.util.module_from_spec(spec)  # type: ignore
                assert spec and spec.loader
                spec.loader.exec_module(mod)  # type: ignore
                MultiAgentReinforcementLearning = getattr(mod, 'MultiAgentReinforcementLearning')
                print(f"[IMPORT] Loaded MultiAgentReinforcementLearning from {c}")
                break
        if MultiAgentReinforcementLearning is None:
            raise ImportError("Could not import MultiAgentReinforcementLearning")

# --------------------------
# Stable-Baselines3 imports
# --------------------------
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor
)
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList, StopTrainingOnRewardThreshold
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy

# --------------------------
# Flatten centralized wrapper
# --------------------------
class FlattenDictWrapper(gym.Wrapper):
    """Flatten Dict observations and actions for centralized PPO control."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.agent_keys = sorted(list(env.observation_space.spaces.keys()))
        self.num_agents = len(self.agent_keys)

        # Flatten observation space
        total_obs_dim = sum(env.observation_space.spaces[k].shape[0] for k in self.agent_keys)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
        )

        # Flatten action space
        act_dim_per_agent = env.action_space.spaces[self.agent_keys[0]].shape[0]
        self._act_dim_per_agent = act_dim_per_agent
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_agents * act_dim_per_agent,), dtype=np.float32
        )

    def _flatten_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        obs_list = [obs_dict[k].astype(np.float32).ravel() for k in self.agent_keys]
        return np.concatenate(obs_list, axis=0)

    def _unflatten_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        action_dict = {}
        for i, k in enumerate(self.agent_keys):
            s = i * self._act_dim_per_agent
            e = (i + 1) * self._act_dim_per_agent
            action_dict[k] = action[s:e]
        return action_dict

    # Gymnasium API
    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs_dict), info

    def step(self, action):
        action_dict = self._unflatten_action(action)
        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self.env.step(action_dict)

        # Sum rewards from all agents
        total_reward = float(sum(reward_dict.values()))

        # Episode ends if ANY agent is done (centralized termination)
        done = bool(any(terminated_dict.values()))
        truncated = bool(any(truncated_dict.values()))
        return self._flatten_obs(obs_dict), total_reward, done, truncated, info_dict

# --------------------------
# Small printer for training metrics
# --------------------------
class RolloutPrinter(BaseCallback):
    """Print moving episode stats from VecMonitor at end of each rollout."""
    def _on_step(self) -> bool:
        # Required by BaseCallback; we don't need per-step logic here.
        return True

    def _on_rollout_end(self) -> None:
        buf = list(self.model.ep_info_buffer)  # VecMonitor fills this
        if buf:
            mean_r = sum(e['r'] for e in buf) / len(buf)
            mean_l = sum(e['l'] for e in buf) / len(buf)
            print(f"[TRAIN] t={self.num_timesteps}  ep_rew_mean~{mean_r:.2f}  ep_len_mean~{mean_l:.1f}")
            self.logger.record("train/ep_rew_mean_recent", mean_r)
            self.logger.record("train/ep_len_mean_recent", mean_l)

class WaypointTrainingCallback(BaseCallback):
    """Collect and log per-episode waypoint metrics when episodes end."""
    def __init__(self, verbose: int = 1, log_freq: int = 100):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_count = 0
        self.last_print = time.time()

        # Stored time series
        self.timesteps = []
        self.total_completed_series = []          # total completions per episode
        self.total_assigned_series = []           # total assignments per episode
        self.total_failed_series = []             # total failures per episode
        self.success_rate_mean_series = []        # mean over agents per episode
        self.per_agent_completed_series = []      # list per episode -> list per agent
        self.coverage_assigned_series = []        # fraction of waypoints assigned at least once
        self.coverage_completed_series = []       # fraction of waypoints completed at least once

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos", [])
        if dones is None:
            return True

        for i, d in enumerate(dones):
            if d:
                self.episode_count += 1
                total_completed = 0
                total_assigned = 0
                total_failed = 0
                per_agent_completed = []
                per_agent_success_rates = []

                if i < len(infos) and infos[i]:
                    for k, v in sorted(infos[i].items()):
                        if k.startswith("agent_") and isinstance(v, dict):
                            c = int(v.get("waypoints_completed", 0))
                            a = int(v.get("waypoints_assigned", 0))
                            f = int(v.get("waypoints_failed", 0))
                            total_completed += c
                            total_assigned += a
                            total_failed += f
                            per_agent_completed.append(c)
                            sr = (c / a) if a > 0 else 0.0
                            per_agent_success_rates.append(sr)

                # Persist series for plotting
                self.timesteps.append(self.num_timesteps)
                self.total_completed_series.append(total_completed)
                self.total_assigned_series.append(total_assigned)
                self.total_failed_series.append(total_failed)
                self.per_agent_completed_series.append(per_agent_completed)
                # Waypoint coverage metrics if env provided waypoint_counts
                coverage_assigned = 0.0
                coverage_completed = 0.0
                wc = infos[i].get('waypoint_counts') if (i < len(infos) and infos[i]) else None
                if isinstance(wc, dict):
                    a_arr = np.asarray(wc.get('assigned', []), dtype=np.int32)
                    c_arr = np.asarray(wc.get('completed', []), dtype=np.int32)
                    if a_arr.size > 0:
                        coverage_assigned = float(np.count_nonzero(a_arr > 0) / a_arr.size)
                    if c_arr.size > 0:
                        coverage_completed = float(np.count_nonzero(c_arr > 0) / c_arr.size)
                self.coverage_assigned_series.append(coverage_assigned)
                self.coverage_completed_series.append(coverage_completed)
                if per_agent_success_rates:
                    self.success_rate_mean_series.append(float(np.mean(per_agent_success_rates)))
                else:
                    self.success_rate_mean_series.append(0.0)

                # Log scalar metrics periodically
                if self.episode_count % self.log_freq == 0:
                    self.logger.record("waypoints/total_completed", total_completed)
                    self.logger.record("waypoints/total_assigned", total_assigned)
                    self.logger.record("waypoints/total_failed", total_failed)
                    if per_agent_completed:
                        self.logger.record("waypoints/mean_completed_per_agent", float(np.mean(per_agent_completed)))
                        self.logger.record("waypoints/std_completed_per_agent", float(np.std(per_agent_completed)))
                    self.logger.record("waypoints/success_rate_mean", self.success_rate_mean_series[-1])
                    self.logger.record("waypoints/coverage_assigned", self.coverage_assigned_series[-1])
                    self.logger.record("waypoints/coverage_completed", self.coverage_completed_series[-1])

                # Throttled print
                if self.verbose and (time.time() - self.last_print) > 10:
                    print(
                        f"[TRAIN] episodes={self.episode_count}  "
                        f"completed_total={total_completed}  assigned_total={total_assigned}  failed_total={total_failed}  "
                        f"success_rate_mean={self.success_rate_mean_series[-1]:.2f}  per_agent_completed={per_agent_completed}"
                    )
                    self.last_print = time.time()
        return True

class RewardTrackerCallback(BaseCallback):
    """Track individual episode rewards for plotting."""
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos", [])
        if dones is None:
            return True

        for i, d in enumerate(dones):
            if d:
                # Extract reward and length from the episode info
                if i < len(infos) and infos[i]:
                    episode_reward = infos[i].get('episode', {}).get('r', 0)
                    episode_length = infos[i].get('episode', {}).get('l', 0)
                    self.episode_rewards.append(float(episode_reward))
                    self.episode_lengths.append(int(episode_length))
                    self.timesteps.append(self.num_timesteps)
        return True

# --------------------------
# Plotting functions
# --------------------------
def plot_rewards(rewards, timesteps, save_path, title="Training Rewards"):
    """Create and save reward progression plot."""
    plt.figure(figsize=(12, 6))

    # Plot rewards
    plt.subplot(1, 2, 1)
    if len(rewards) > 0:
        plt.plot(timesteps, rewards, 'b-', alpha=0.7, linewidth=1)
        plt.title(f'{title} - Rewards')
    else:
        plt.text(0.5, 0.5, 'No episode rewards recorded', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{title} - Rewards (empty)')
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.grid(True, alpha=0.3)

    # Plot moving average
    if len(rewards) > 10:
        window_size = min(50, len(rewards) // 10)
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        moving_timesteps = timesteps[window_size-1:]
        plt.plot(moving_timesteps, moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')
        plt.legend()

    # Plot reward distribution
    plt.subplot(1, 2, 2)
    if len(rewards) > 0:
        plt.hist(rewards, bins=30, alpha=0.7, color='blue', edgecolor='black')
    else:
        plt.text(0.5, 0.5, 'No data', ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title(f'{title} - Reward Distribution')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'reward_progression.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Reward progression plot saved to {os.path.join(save_path, 'reward_progression.png')}")

def plot_reward_stats(rewards, timesteps, save_path):
    """Create statistical plots of rewards."""
    plt.figure(figsize=(15, 10))

    # Rolling statistics
    plt.subplot(2, 2, 1)
    window = min(100, len(rewards) // 5)
    if len(rewards) > window and window > 0:
        rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
        rolling_std = [np.std(rewards[i:i+window]) for i in range(len(rewards)-window+1)]
        rolling_timesteps = timesteps[window-1:]

        plt.plot(rolling_timesteps, rolling_mean, 'b-', label='Rolling Mean')
        plt.fill_between(rolling_timesteps,
                        np.array(rolling_mean) - np.array(rolling_std),
                        np.array(rolling_mean) + np.array(rolling_std),
                        alpha=0.3, color='blue', label='±1 Std Dev')
        plt.xlabel('Timesteps')
        plt.ylabel('Rolling Reward Stats')
        plt.title(f'Rolling Statistics (window={window})')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Cumulative best reward
    plt.subplot(2, 2, 2)
    if len(rewards) > 0:
        best_rewards = []
        current_best = float('-inf')
        for r in rewards:
            current_best = max(current_best, r)
            best_rewards.append(current_best)
        plt.plot(timesteps, best_rewards, 'g-', linewidth=2)
        plt.xlabel('Timesteps')
        plt.ylabel('Best Reward So Far')
        plt.title('Cumulative Best Reward')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No data', ha='center', va='center', transform=plt.gca().transAxes)

    # Reward improvement rate
    plt.subplot(2, 2, 3)
    if len(rewards) > 10:
        improvements = np.diff(rewards)
        plt.plot(timesteps[1:], improvements, 'r-', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Timesteps')
        plt.ylabel('Reward Change')
        plt.title('Episode-to-Episode Reward Changes')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No data', ha='center', va='center', transform=plt.gca().transAxes)

    # Learning curves comparison
    plt.subplot(2, 2, 4)
    if len(rewards) > 20:
        # Split into thirds for comparison
        third = len(rewards) // 3
        early_rewards = rewards[:third]
        mid_rewards = rewards[third:2*third]
        late_rewards = rewards[2*third:]

        plt.boxplot([early_rewards, mid_rewards, late_rewards],
                   labels=['Early', 'Middle', 'Late'])
        plt.ylabel('Episode Reward')
        plt.title('Reward Distribution by Training Phase')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'reward_statistics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Reward statistics plot saved to {os.path.join(save_path, 'reward_statistics.png')}")

def plot_waypoint_metrics(timesteps, total_completed, per_agent_completed_series,
                          total_assigned, total_failed, success_rate_mean,
                          save_path, title="Waypoint Metrics"):
    """Plot waypoint-related training metrics and save as PNG."""
    # Always produce a figure (even if empty) to satisfy 'always create graphs'

    # Prepare x-axis
    x = np.array(timesteps) if len(timesteps) > 0 else np.array([0])

    plt.figure(figsize=(14, 10))

    # (1) Total completions over time
    plt.subplot(2, 2, 1)
    if len(total_completed) > 0:
        plt.plot(x if len(timesteps)>0 else range(len(total_completed)), total_completed, 'g-', label='Total completions/ep')
    if len(total_completed) > 10:
        w = max(5, len(total_completed)//20)
        ma = np.convolve(total_completed, np.ones(w)/w, mode='valid')
        xx = (x if len(timesteps)>0 else np.arange(len(total_completed)))[w-1:]
        plt.plot(xx, ma, 'k--', alpha=0.8, label=f'MA (w={w})')
    plt.xlabel('Timesteps'); plt.ylabel('Completions'); plt.title('Waypoint Completions'); plt.grid(True, alpha=0.3); plt.legend()

    # (2) Success rate (mean across agents)
    plt.subplot(2, 2, 2)
    sr = np.array(success_rate_mean) if len(success_rate_mean) > 0 else np.array([])
    if sr.size > 0:
        plt.plot(x if len(timesteps)>0 else range(len(sr)), sr, 'b-')
    else:
        plt.text(0.5, 0.5, 'No data', ha='center', va='center', transform=plt.gca().transAxes)
    plt.ylim(0.0, 1.05)
    plt.xlabel('Timesteps'); plt.ylabel('Success rate'); plt.title('Mean Success Rate'); plt.grid(True, alpha=0.3)

    # (3) Per-agent completions (lines)
    plt.subplot(2, 2, 3)
    if per_agent_completed_series and per_agent_completed_series[0]:
        n_agents = len(per_agent_completed_series[0])
        for a in range(n_agents):
            series = [row[a] if a < len(row) else 0 for row in per_agent_completed_series]
            plt.plot(x if len(timesteps)>0 else range(len(series)), series, label=f'agent_{a}')
        plt.legend(ncol=2, fontsize=8)
    plt.xlabel('Timesteps'); plt.ylabel('Completions'); plt.title('Per-Agent Completions'); plt.grid(True, alpha=0.3)

    # (4) Assigned vs failed per episode
    plt.subplot(2, 2, 4)
    if len(total_assigned) > 0:
        plt.plot(x if len(timesteps)>0 else range(len(total_assigned)), total_assigned, 'c-', label='Assigned/ep')
    if len(total_failed) > 0:
        plt.plot(x if len(timesteps)>0 else range(len(total_failed)), total_failed, 'r-', label='Failed/ep')
    plt.xlabel('Timesteps'); plt.ylabel('Count'); plt.title('Assigned vs Failed'); plt.grid(True, alpha=0.3); plt.legend()

    plt.tight_layout()
    out_path = os.path.join(save_path, 'waypoint_metrics.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Waypoint metrics plot saved to {out_path}")

# --------------------------
# Env factory
# --------------------------
def make_env(args, rank: int = 0):
    """Returns a thunk that creates ONE raw env -> FlattenDictWrapper -> Monitor."""
    def _init():
        env = MultiAgentReinforcementLearning(
            num_drones=args.num_drones,
            num_waypoints=args.num_waypoints,
            episode_len_sec=args.episode_len,
            waypoint_radius=args.waypoint_radius,
            waypoint_hold_steps=args.waypoint_hold_steps,
            priority_mode=args.priority_mode,
            min_waypoint_separation=args.min_waypoint_separation,
            terminate_when_pool_exhausted=(not args.no_pool_exhaustion_termination),
            gui=False,  # GUI off for training
            verbose=args.verbose,
            pyb_freq=args.pyb_freq,
            ctrl_freq=args.ctrl_freq
        )
        env = FlattenDictWrapper(env)
        # Per-env Monitor (OK together with VecMonitor at the vector level)
        env = Monitor(env)
        if args.seed is not None:
            env.reset(seed=args.seed + rank)
        return env
    return _init

# --------------------------
# CLI
# --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on dynamic waypoint environment (centralized)")

    # Training
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # PPO
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    # Env
    p.add_argument("--num-drones", type=int, default=4)
    p.add_argument("--num-waypoints", type=int, default=20)
    p.add_argument("--episode-len", type=int, default=3000)
    p.add_argument("--waypoint-radius", type=float, default=0.5)
    p.add_argument("--waypoint-hold-steps", type=int, default=5)
    p.add_argument("--priority-mode", type=str, default="distance",
                   choices=["sequential", "distance", "random"])
    p.add_argument("--min-waypoint-separation", type=float, default=2.0)
    p.add_argument("--pyb-freq", type=int, default=240)
    p.add_argument("--ctrl-freq", type=int, default=60)
    # Early termination
    p.add_argument("--no-pool-exhaustion-termination", action="store_true",
                   help="Disable early termination when the waypoint pool is exhausted")

    # Output
    p.add_argument("--save-path", type=str, default="models/dynamic_waypoints_ppo")
    p.add_argument("--log-dir", type=str, default="logs/")
    p.add_argument("--save-freq", type=int, default=50_000)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--target-reward", type=float, default=100.0, help="Target reward for early stopping")
    p.add_argument("--early-stop-on-reward", action="store_true", help="Stop when average eval reward >= target")
    p.add_argument("--eval-deterministic", action="store_true", help="Use deterministic policy for evaluation")

    # Quick smoke test
    p.add_argument("--quick", action="store_true", help="Reduced settings for a fast test run")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging for debugging")
    # Always plotting; legacy flags removed

    return p.parse_args()

# --------------------------
# main
# --------------------------
def main():
    args = parse_args()

    # Quick mode
    if args.quick:
        args.timesteps = 50_000
        args.n_steps = 10*512
        args.eval_freq = 5_000
        args.save_freq = 5_000
        args.num_waypoints = 20
        print("[INFO] Quick mode enabled")

    # Seeds
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # IO
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ---------- Build TRAIN env: Dummy/Subproc -> VecMonitor -> VecNormalize ----------
    print(f"Creating {args.n_envs} training environment(s)...")
    if args.n_envs == 1:
        train_env: VecNormalize = DummyVecEnv([make_env(args, rank=0)])
    else:
        train_env = SubprocVecEnv([make_env(args, rank=i) for i in range(args.n_envs)])
    train_env = VecMonitor(train_env)  # episode stats at vector level
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ---------- Build EVAL env: Dummy -> VecMonitor -> VecNormalize (no reward norm) ----------
    print("Creating evaluation environment...")
    eval_env: VecNormalize = DummyVecEnv([make_env(args, rank=1_000)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    # Sync eval stats with train (and EvalCallback will keep syncing)
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms

    # ---------- Build EVAL env: Dummy -> VecMonitor -> VecNormalize (no reward norm) ----------
    # Evaluation has been removed to keep training uninterrupted.

    # ---------- PPO model ----------
    #policy_kwargs = dict(net_arch=[512, 512, 256])

    policy_kwargs = dict(
    activation_fn=nn.ReLU,
    net_arch=dict(pi=[512,512,256], vf=[512,512,256]),
    log_std_init=-0.8,
    ortho_init=True,
    share_features_extractor=False,
)
    
    print("Creating PPO model.")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        verbose=0,  # we’ll print via logger + callbacks
        tensorboard_log=args.log_dir,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device=args.device
    )

    # Logger -> stdout + csv + tensorboard (shows rollout/ep_rew_mean etc.)
    new_logger = configure(args.log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # ---------- Callbacks ----------
    callbacks = []

    # Reward tracker for plotting
    reward_tracker = RewardTrackerCallback(verbose=0)
    callbacks.append(reward_tracker)

    # Rolling train printer + custom waypoint metrics
    callbacks.append(RolloutPrinter())
    waypoint_cb = WaypointTrainingCallback(verbose=1, log_freq=100)
    callbacks.append(waypoint_cb)

    # Checkpoints (also save VecNormalize)
    callbacks.append(CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.save_path,
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True
    ))

    # Periodic evaluation during training (DURING). Saves best_model.zip
    callback_on_best = None
    if args.early_stop_on_reward:
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=args.target_reward, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        best_model_save_path=args.save_path,
        log_path=args.save_path,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=args.eval_deterministic,
        render=False
    )
    callbacks.append(eval_cb)

    # ---------- Train ----------
    print("\n" + "="*60)
    print("Starting training with configuration:")
    print(f"  Drones: {args.num_drones}")
    print(f"  Waypoints: {args.num_waypoints}")
    print(f"  Priority mode: {args.priority_mode}")
    print(f"  Timesteps: {args.timesteps}")
    print(f"  LR: {args.lr} | n_steps: {args.n_steps} | batch: {args.batch_size}")
    print("="*60 + "\n")

    try:
        model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks), progress_bar=True)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")

    # ---------- Save ----------
    final_path = os.path.join(args.save_path, "final_model")
    model.save(final_path)
    # Save final VecNormalize stats to reuse at inference time
    train_env.save(os.path.join(args.save_path, "vecnormalize.pkl"))
    print(f"\n[OK] Final model saved to {final_path}.zip")
    print(f"[OK] VecNormalize stats saved to {os.path.join(args.save_path, 'vecnormalize.pkl')}")

    # ---------- Generate plots (always) ----------
    print("\n[PLOT] Generating reward and waypoint plots...")
    plot_rewards(reward_tracker.episode_rewards, reward_tracker.timesteps,
                 args.save_path, "MARL Dynamic Waypoints PPO")
    plot_reward_stats(reward_tracker.episode_rewards, reward_tracker.timesteps,
                      args.save_path)

    # Save raw reward data (possibly empty)
    reward_data = {
        'timesteps': np.array(reward_tracker.timesteps, dtype=np.int64),
        'rewards': np.array(reward_tracker.episode_rewards, dtype=np.float32),
        'lengths': np.array(reward_tracker.episode_lengths, dtype=np.int32)
    }
    np.savez(os.path.join(args.save_path, 'reward_data.npz'), **reward_data)
    print(f"[SAVE] Raw reward data saved to {os.path.join(args.save_path, 'reward_data.npz')}")

    # Waypoint metrics plots (always create a figure)
    plot_waypoint_metrics(
        timesteps=waypoint_cb.timesteps,
        total_completed=waypoint_cb.total_completed_series,
        per_agent_completed_series=waypoint_cb.per_agent_completed_series,
        total_assigned=waypoint_cb.total_assigned_series,
        total_failed=waypoint_cb.total_failed_series,
        success_rate_mean=waypoint_cb.success_rate_mean_series,
        save_path=args.save_path,
        title="MARL Waypoint Metrics"
    )

    # Save raw waypoint metrics (possibly empty)
    max_agents = max((len(row) for row in waypoint_cb.per_agent_completed_series), default=0)
    per_agent_arr = np.zeros((len(waypoint_cb.per_agent_completed_series), max_agents), dtype=np.int32)
    for idx, row in enumerate(waypoint_cb.per_agent_completed_series):
        for j, val in enumerate(row):
            per_agent_arr[idx, j] = val

    wp_metrics = {
        'timesteps': np.array(waypoint_cb.timesteps, dtype=np.int64),
        'total_completed': np.array(waypoint_cb.total_completed_series, dtype=np.int32),
        'total_assigned': np.array(waypoint_cb.total_assigned_series, dtype=np.int32),
        'total_failed': np.array(waypoint_cb.total_failed_series, dtype=np.int32),
        'success_rate_mean': np.array(waypoint_cb.success_rate_mean_series, dtype=np.float32),
        'per_agent_completed': per_agent_arr
    }
    np.savez(os.path.join(args.save_path, 'waypoint_metrics.npz'), **wp_metrics)
    print(f"[SAVE] Waypoint metrics saved to {os.path.join(args.save_path, 'waypoint_metrics.npz')}")

    # Cleanup
    train_env.close()

if __name__ == "__main__":
    main()
