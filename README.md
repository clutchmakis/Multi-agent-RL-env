Multi-Agent RL Environment: Dynamic Waypoints
=============================================

A lightweight research sandbox for multi‑agent navigation with dynamic waypoint allocation, built on top of Gymnasium + PyBullet + Stable‑Baselines3. It includes:

- A custom multi‑agent env with a shared waypoint pool and assignment/coordination logic.
- Centralized PPO training (flattened multi‑agent observations/actions) with TensorBoard logging.
- Rich metrics and plots (rewards, waypoint coverage/success, per‑agent stats) saved automatically.
- A GUI visualizer for trained models with slow‑motion and on‑screen debug overlay.


Repo Layout
-----------
- `env/Marl_dynamic_waypoints.py`: Main environment (dynamic waypoint pool, assignment, rewards, early termination).
- `env/BaseRLAviary.py`, `env/BaseAviary.py`: Minimal aviary scaffolding used by the env.
- `examples/Marl_dyn_train.py.py`: Centralized PPO training script + plotting + best‑model eval callback.
- `examples/Marl_dyn_vis.py`: GUI visualizer for trained models (or random/heuristic with minor changes).
- `utils/*`: Assorted helpers and enums.


Key Features
------------
- Dynamic waypoint pool with minimum separation and optional reuse of completed waypoints.
- Multiple assignment modes: `sequential`, `distance`, `random`. If SciPy is available, a Hungarian/global assignment pass is used for initial allocations.
- Blended control: learned 3D action is blended with the waypoint direction and sent to the PID controller.
- Rich info dict: per‑agent completions/assignments/failures; per‑waypoint assigned/completed/failed counts for the current episode.
- Early termination (pool exhaustion only): episode ends after ALL waypoints are completed and no agent has an active target; a short grace window keeps the sim running a few extra steps for visibility.


Requirements
------------
- Python 3.9+
- `pip install "stable-baselines3[extra]" gymnasium pybullet matplotlib numpy`
- Optional: `scipy` (for Hungarian/global assignment)
- Optional: `tensorboard` (to view training logs)


Quick Start (Training)
----------------------
- Train a small model and auto‑generate plots and logs:
  - `python3 examples/Marl_dyn_train.py.py --quick`
- Typical training run (single env):
  - `python3 examples/Marl_dyn_train.py.py --timesteps 500000 --num-drones 4 --num-waypoints 20`
- Multi‑process vector envs:
  - `--n-envs 4` (ensure your machine can handle the extra processes)

Outputs (under `--save-path`, default `models/dynamic_waypoints_ppo`):
- `final_model.zip`: last checkpoint
- `best_model.zip`: best eval return during training
- `vecnormalize.pkl`: observation/reward normalization stats
- Plots + data: `reward_progression.png`, `reward_statistics.png`, `reward_data.npz`, `waypoint_metrics.png`, `waypoint_metrics.npz`

TensorBoard (under `--log-dir`, default `logs/`):
- Run: `tensorboard --logdir logs/`
- Scalars include `train/ep_rew_mean_recent` and `waypoints/*` (total_assigned, total_completed, total_failed, success_rate_mean, coverage_*).


Visualizer (GUI)
----------------
- Run a trained model with GUI:
  - `python3 examples/Marl_dyn_vis.py --model models/dynamic_waypoints_ppo/final_model.zip --episodes 3`
- Useful flags:
  - `--slow-motion --step-delay 0.05` (easier to inspect behavior)
  - `--debug-overlay --highlight-lines` (draws per‑drone text and a line to its target)
  - `--out-dir examples` (where plots/npz are saved)
- Plots saved: `vis_episode_metrics.png` (returns/completions), `vis_waypoint_metrics.png` (success rate, completions/min); raw metrics: `vis_metrics.npz`.


Environment Details
-------------------
- Observation/Action:
  - Multi‑agent dict observation is flattened for centralized PPO via a thin wrapper.
  - Action is a 3‑vector per agent; env blends policy action with waypoint direction and feeds a PID controller.
- Waypoints:
  - Generated in a bounded workspace with minimum separation.
  - Completed waypoints are retired (no reuse).
- Assignment:
  - Modes: `sequential` (first available), `distance` (closest), `random`.
  - If SciPy is installed, a global assignment (Hungarian) is used for initial assignment.
- Early Termination (pool exhaustion):
  - Triggers only when ALL waypoints are in COMPLETED state and no drones have targets.
  - A short grace window (a few env steps) is applied after the final completion so you see the result in GUI.


Common CLI Flags (Training)
---------------------------
- `--timesteps`: total training steps
- `--n-envs`: number of parallel envs
- `--num-drones`, `--num-waypoints`
- `--waypoint-radius`, `--waypoint-hold-steps`: completion criteria
- `--priority-mode`: `sequential|distance|random`
  (Completed waypoints are always retired; no reuse option.)
- `--pyb-freq`, `--ctrl-freq`: physics/control frequencies
- `--save-path`, `--log-dir`, `--save-freq`, `--eval-freq`, `--eval-episodes`
- `--device`: `cpu|cuda|auto`


Metrics & Files
---------------
- Training
  - Rewards over timesteps + moving averages + distributions
  - Waypoints: per‑episode totals (assigned/completed/failed), per‑agent completions, mean success rate (completed/assigned), coverage (assigned/completed at least once)
  - Files: `reward_progression.png`, `reward_statistics.png`, `reward_data.npz`, `waypoint_metrics.png`, `waypoint_metrics.npz`
- Visualization
  - Episode returns, total completions, success rate, completions per minute
  - Files: `vis_episode_metrics.png`, `vis_waypoint_metrics.png`, `vis_metrics.npz`


Debugging Tips
--------------
- Use `--slow-motion --step-delay 0.05 --debug-overlay --highlight-lines` in the visualizer to see targets, hold counters, and distances.
- If TensorBoard curves are flat, check that the obs dim in training matches the visualizer (the scripts already sanity‑check this).
- SciPy not installed? The env falls back to priority mode; you can still train/run.


Troubleshooting
---------------
- PyBullet warnings like “Removing body failed”: suppressed by avoiding redundant remove calls; if you add custom visuals, prefer clearing your own lists rather than removing bodies manually on reset.
- Early termination happens too soon: remember it triggers only when all waypoints are completed and no targets remain; a grace window keeps the sim visible for a few extra steps.
- Obs mismatch between train/vis: make sure both scripts import the same `env/Marl_dynamic_waypoints.py` (they try local fallback first).


Acknowledgements
----------------
- Built on top of PyBullet and Stable‑Baselines3.
- Inspired by drone aviary environments (modularized here for this project).

