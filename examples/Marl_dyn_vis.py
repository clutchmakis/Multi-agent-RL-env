"""
Visualization script for dynamic waypoint environment.

- Works with trained PPO models (using the SAME VecNormalize stats saved during training)
- Or with heuristic / random policies
- GUI enabled, static waypoint colors (no per-step recoloring)
- Sanity-checks model vs env observation size to avoid (1, 328) vs (1, 508) errors
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# -----------------------------------------------------------------------------
# Import environment (match training import order: package first)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# SB3 (for trained model)
# -----------------------------------------------------------------------------
HAS_SB3 = True
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except Exception:
    HAS_SB3 = False

# -----------------------------------------------------------------------------
# PyBullet
# -----------------------------------------------------------------------------
try:
    import pybullet as p
except ImportError:
    print("PyBullet not installed. Install with: pip install pybullet")
    raise


# -----------------------------------------------------------------------------
# Flatten centralized wrapper (same as in training)
# -----------------------------------------------------------------------------
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
        total_reward = float(sum(reward_dict.values()))
        done = bool(any(terminated_dict.values()))
        truncated = bool(any(truncated_dict.values()))
        return self._flatten_obs(obs_dict), total_reward, done, truncated, info_dict



# -----------------------------------------------------------------------------
# Visualizer
# -----------------------------------------------------------------------------
class DroneVisualizer:
    def __init__(self, args):
        self.args = args
        self.using_model = bool(args.model and HAS_SB3)
        self.vec = None               # Vec env (model path)
        self.env = None               # Raw env (heuristic/random path)
        self.policy = None
        self.wrapped_env = None       # Flattened wrapper for random path (action sampling)
        self.episode_stats = []
        self._debug_last_ids = []     # store debug text/line ids when overlay enabled

        if self.using_model:
            self._setup_model_path()
        else:
            self._setup_non_model_path()

    # ---------- Setup for trained model ----------
    def _setup_model_path(self):
        """Create a vectorized GUI env, load VecNormalize stats, load the model."""
        def _make_gui_env():
            e = MultiAgentReinforcementLearning(
                num_drones=self.args.num_drones,
                num_waypoints=self.args.num_waypoints,
                episode_len_sec=self.args.episode_len,
                waypoint_radius=self.args.waypoint_radius,
                waypoint_hold_steps=self.args.waypoint_hold_steps,
                priority_mode=self.args.priority_mode,
                reuse_completed_waypoints=self.args.reuse_completed_waypoints,
                min_waypoint_separation=self.args.min_waypoint_separation,
                gui=True,  # GUI ON
                record=self.args.record,
                verbose= True, #self.args.verbose,
                pyb_freq=self.args.pyb_freq,
                ctrl_freq=self.args.ctrl_freq,
                visualize_waypoints=True,
                waypoint_marker_radius=0.15
            )
            # Fast stepping
            #p.setRealTimeSimulation(0, physicsClientId=e.CLIENT)
            # Camera + GUI toggles
            self._setup_camera(client_id=e.CLIENT)
            # Static color mode already in env (we removed per-step recolors)
            return FlattenDictWrapper(e)

        # One-env vector for inference
        self.vec = DummyVecEnv([_make_gui_env])

        # Load VecNormalize stats saved at training time (same directory as model)
        model_dir = os.path.dirname(os.path.abspath(self.args.model))
        stats_path = os.path.join(model_dir, "vecnormalize.pkl")
        if os.path.exists(stats_path):
            try:
                self.vec = VecNormalize.load(stats_path, self.vec)
                self.vec.training = False
                self.vec.norm_reward = False
                print(f"[VIS] Loaded VecNormalize stats: {stats_path}")
            except Exception as e:
                print(f"[VIS] Warning: failed to load VecNormalize stats: {e}")
        else:
            print(f"[VIS] Warning: VecNormalize stats not found at {stats_path}. "
                  "If you trained with normalization, behavior may differ.")

        if not HAS_SB3:
            raise RuntimeError("stable-baselines3 not installed; cannot load a model.")

        # Load model
        self.policy = PPO.load(self.args.model, device=self.args.device)
        print(f"[VIS] Loaded model from {self.args.model}")

        # ---- Sanity check: env obs dim vs model expected dim ----
        try:
            obs = self.vec.reset()
            curr_dim = int(obs.shape[1])  # (n_env=1, dim)
        except Exception:
            # Some wrappers return tuple (obs, info)
            obs, _info = self.vec.reset(), None
            curr_dim = int(obs.shape[1])
        expected_dim = int(self.policy.observation_space.shape[0])

        if curr_dim != expected_dim:
            # Try to print the env module file path to diagnose mismatched env copy
            env_mod_name = MultiAgentReinforcementLearning.__module__
            env_file = None
            try:
                env_file = sys.modules[env_mod_name].__file__
            except Exception:
                pass
            msg = (f"\n[VIS][ERROR] Observation size mismatch:\n"
                   f"  Env (GUI) flattened obs dim = {curr_dim}\n"
                   f"  Model expects               = {expected_dim}\n"
                   f"  Env module: {env_mod_name}  file: {env_file}\n"
                   "This usually means the visualizer imported a different copy of "
                   "Marl_dynamic_waypoints.py than the one used for training.\n"
                   "Fix: ensure the SAME env file is imported (this script prefers the local file "
                   "next to it). Also confirm _setupMultiAgentSpaces computes obs_dim dynamically.")
            raise RuntimeError(msg)


    # ---------- Camera / GUI ----------
    def _setup_camera(self, client_id: int):
        """Setup PyBullet camera and optional GUI/shadow toggles."""
        p.resetDebugVisualizerCamera(
            cameraDistance=8.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 1],
            physicsClientId=client_id
        )
        if getattr(self.args, "no_gui_panels", False):
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client_id)
        if getattr(self.args, "no_shadows", False):
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=client_id)

    # ---------- Waypoint color updates: NO-OP for speed ----------
    def _update_waypoint_colors(self):
        return

    # ---------- Episode loop ----------
    def run_episode(self, episode_idx: int):
        """Run a single episode in GUI, return stats dict."""
        step = 0
        ep_reward = 0.0
        last_completions = {}
        last_assigned = {}
        last_failed = {}

        if self.using_model:
            obs = self.vec.reset()
            done = False
            while not done:
                action, _ = self.policy.predict(obs, deterministic=True)
                obs, rewards, dones, infos = self.vec.step(action)
                r = float(rewards[0])
                ep_reward += r
                done = bool(dones[0])
                step += 1

                if infos and infos[0]:
                    info0 = infos[0]
                    for k, v in info0.items():
                        if isinstance(v, dict) and k.startswith("agent_"):
                            last_completions[k] = int(v.get("waypoints_completed", last_completions.get(k, 0)))
                            last_assigned[k] = int(v.get("waypoints_assigned", last_assigned.get(k, 0)))
                            last_failed[k] = int(v.get("waypoints_failed", last_failed.get(k, 0)))

                if self.args.slow_motion:
                    time.sleep(self.args.step_delay)

                # Debug overlay: annotate drones and highlight target lines
                if getattr(self.args, 'debug_overlay', False):
                    try:
                        self._draw_debug_overlay()
                    except Exception as e:
                        print(f"[VIS][DEBUG] overlay error: {e}")
        

        total_completions = int(sum(last_completions.values())) if last_completions else 0
        per_agent = [last_completions.get(f"agent_{i}", 0) for i in range(self.args.num_drones)]
        per_agent_assigned = [last_assigned.get(f"agent_{i}", 0) for i in range(self.args.num_drones)]
        per_agent_failed = [last_failed.get(f"agent_{i}", 0) for i in range(self.args.num_drones)]
        success_rates = [ (c/a) if a > 0 else 0.0 for c, a in zip(per_agent, per_agent_assigned) ]
        mean_success = float(np.mean(success_rates)) if success_rates else 0.0
        duration_sec = step / float(self.args.ctrl_freq) if self.args.ctrl_freq > 0 else 0.0
        completions_per_min = (total_completions / duration_sec * 60.0) if duration_sec > 0 else 0.0
        stats = {
            "episode": episode_idx,
            "reward": ep_reward,
            "total_completions": total_completions,
            "per_agent": per_agent,
            "per_agent_assigned": per_agent_assigned,
            "per_agent_failed": per_agent_failed,
            "success_rate_mean": mean_success,
            "steps": step,
            "duration_sec": duration_sec,
            "completions_per_min": completions_per_min
        }

        print(f"[EP {episode_idx}] return={ep_reward:.2f}  "
              f"completed_total={total_completions}  per_agent={per_agent}")
        return stats

    def _draw_debug_overlay(self):
        """Draw per-drone text with target, hold, dist, and a line to target."""
        # Remove previous items (lifeTime would auto-expire but keep tidy if step_delay large)
        for uid in self._debug_last_ids:
            try:
                p.removeUserDebugItem(uid)
            except Exception:
                pass
        self._debug_last_ids = []

        # Access underlying env
        client_id = self.vec.get_attr('CLIENT')[0]
        agent_targets = self.vec.get_attr('agent_target_waypoint')[0]
        holds = self.vec.get_attr('agent_hold_counter')[0]
        waypoint_pool = self.vec.get_attr('waypoint_pool')[0]

        n = len(agent_targets)
        for i in range(n):
            # Drone pos
            state = self.vec.env_method('_getDroneStateVector', i)[0]
            pos = np.array(state[0:3])
            tgt_idx = int(agent_targets[i])
            hold = int(holds[i])
            dist = None
            tgt_pos = None
            if 0 <= tgt_idx < waypoint_pool.shape[0]:
                tgt_pos = waypoint_pool[tgt_idx]
                dist = float(np.linalg.norm(pos - tgt_pos))
            # Text above drone
            txt = f"A{i} tgt={tgt_idx} hold={hold}" + (f" d={dist:.2f}" if dist is not None else "")
            uid = p.addUserDebugText(txt, pos + np.array([0,0,0.4]),
                                     textColorRGB=[1,1,0],
                                     textSize=1.2,
                                     lifeTime=self.args.step_delay if self.args.slow_motion else 0.03,
                                     physicsClientId=client_id)
            self._debug_last_ids.append(uid)

            # Line to target
            if getattr(self.args, 'highlight_lines', False) and tgt_pos is not None:
                uid2 = p.addUserDebugLine(pos, tgt_pos, [0,1,0], lineWidth=2,
                                          lifeTime=self.args.step_delay if self.args.slow_motion else 0.03,
                                          physicsClientId=client_id)
                self._debug_last_ids.append(uid2)

    # ---------- Multi-episode driver ----------
    def run(self):
        print(f"\nStarting visualization for {self.args.episodes} episode(s)")
        print(f"  Drones: {self.args.num_drones} | Waypoints: {self.args.num_waypoints} | "
              f"Priority: {self.args.priority_mode} | Reuse: {self.args.reuse_completed_waypoints}")
        try:
            for ep in range(self.args.episodes):
                st = self.run_episode(ep)
                self.episode_stats.append(st)
                if ep < self.args.episodes - 1 and self.args.pause_between:
                    input("Press Enter for next episode...")
        except KeyboardInterrupt:
            print("\n[VIS] Interrupted by user")
        finally:
            if self.using_model and self.vec is not None:
                self.vec.close()
            if self.env is not None:
                self.env.close()

            if self.episode_stats:
                rewards = [s["reward"] for s in self.episode_stats]
                totals = [s["total_completions"] for s in self.episode_stats]
                print("\n" + "="*54)
                print("Overall statistics")
                print("="*54)
                print(f"Avg return: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
                print(f"Avg completions: {np.mean(totals):.2f} ± {np.std(totals):.2f}")
                print(f"Max completions: {max(totals)} | Min: {min(totals)}")

                # Save plots and raw data
                out_dir = getattr(self.args, 'out_dir', CURRENT_DIR)
                os.makedirs(out_dir, exist_ok=True)

                # Plot rewards per episode
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(rewards, 'b-o', alpha=0.8)
                plt.xlabel('Episode'); plt.ylabel('Return'); plt.title('Return per Episode'); plt.grid(True, alpha=0.3)

                # Plot completions per episode
                plt.subplot(1, 2, 2)
                plt.plot(totals, 'g-o', alpha=0.8)
                plt.xlabel('Episode'); plt.ylabel('Total Completions'); plt.title('Completions per Episode'); plt.grid(True, alpha=0.3)
                out_path = os.path.join(out_dir, 'vis_episode_metrics.png')
                plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
                print(f"[VIS][PLOT] Saved {out_path}")

                # Detailed waypoint metrics if available
                suc = [s.get("success_rate_mean", 0.0) for s in self.episode_stats]
                cpm = [s.get("completions_per_min", 0.0) for s in self.episode_stats]
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(suc, 'm-o'); plt.ylim(0, 1.05)
                plt.xlabel('Episode'); plt.ylabel('Mean Success Rate'); plt.title('Success Rate per Episode'); plt.grid(True, alpha=0.3)
                plt.subplot(1, 2, 2)
                plt.plot(cpm, 'c-o')
                plt.xlabel('Episode'); plt.ylabel('Completions/min'); plt.title('Completions per Minute'); plt.grid(True, alpha=0.3)
                out_path2 = os.path.join(out_dir, 'vis_waypoint_metrics.png')
                plt.tight_layout(); plt.savefig(out_path2, dpi=150, bbox_inches='tight'); plt.close()
                print(f"[VIS][PLOT] Saved {out_path2}")

                # Save raw data
                # Pad per-agent arrays to max agent count
                max_agents = max((len(s.get('per_agent', [])) for s in self.episode_stats), default=0)
                per_agent = np.zeros((len(self.episode_stats), max_agents), dtype=np.int32)
                per_agent_assigned = np.zeros_like(per_agent)
                per_agent_failed = np.zeros_like(per_agent)
                for i, s in enumerate(self.episode_stats):
                    row = s.get('per_agent', [])
                    row_a = s.get('per_agent_assigned', [])
                    row_f = s.get('per_agent_failed', [])
                    for j in range(min(max_agents, len(row))):
                        per_agent[i, j] = row[j]
                    for j in range(min(max_agents, len(row_a))):
                        per_agent_assigned[i, j] = row_a[j]
                    for j in range(min(max_agents, len(row_f))):
                        per_agent_failed[i, j] = row_f[j]
                out_npz = os.path.join(out_dir, 'vis_metrics.npz')
                np.savez(out_npz,
                         rewards=np.array(rewards, dtype=np.float32),
                         total_completions=np.array(totals, dtype=np.int32),
                         per_agent_completed=per_agent,
                         per_agent_assigned=per_agent_assigned,
                         per_agent_failed=per_agent_failed,
                         success_rate_mean=np.array(suc, dtype=np.float32),
                         completions_per_min=np.array(cpm, dtype=np.float32))
                print(f"[VIS][SAVE] Saved raw metrics to {out_npz}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Visualize dynamic waypoint navigation")

    # Policy/Model selection
    p.add_argument("--model", type=str, default=None,
                   help="Path to trained model .zip (expects vecnormalize.pkl next to it)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    

    # Env params (must match training for best results)
    p.add_argument("--num-drones", type=int, default=4)
    p.add_argument("--num-waypoints", type=int, default=20)
    p.add_argument("--episode-len", type=int, default=60)
    p.add_argument("--waypoint-radius", type=float, default=0.5)
    p.add_argument("--waypoint-hold-steps", type=int, default=5)
    p.add_argument("--priority-mode", type=str, default="distance",
                   choices=["sequential", "distance", "random"])
    p.add_argument("--reuse-completed-waypoints", action="store_true")
    p.add_argument("--min-waypoint-separation", type=float, default=2.0)
    p.add_argument("--pyb-freq", type=int, default=240)
    p.add_argument("--ctrl-freq", type=int, default=60)

    # Visualization
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--slow-motion", action="store_true")
    p.add_argument("--step-delay", type=float, default=0.01)
    p.add_argument("--pause-between", action="store_true",
                   help="Pause between episodes and wait for Enter")
    p.add_argument("--no-gui-panels", action="store_true")
    p.add_argument("--no-shadows", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--record", action="store_true")
    p.add_argument("--debug-overlay", action="store_true", help="Draw per-drone debug text and target lines")
    p.add_argument("--highlight-lines", action="store_true", help="Draw a line from drone to its target waypoint")
    p.add_argument("--out-dir", type=str, default=CURRENT_DIR, help="Directory to save plots & metrics")

    return p.parse_args()


def main():
    args = parse_args()
    if args.model is not None and not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    vis = DroneVisualizer(args)
    vis.run()


if __name__ == "__main__":
    main()
