Dynamic Waypoints MARL — Detailed Design and Guide
=================================================

This document explains the architecture, environment dynamics, training and visualization pipelines, and the metrics produced by this codebase. It also includes a function‑level reference for the environment and guidance on extending or debugging the system.


1) Big‑Picture Overview
-----------------------
- Multi‑agent drones must visit a shared pool of 3D waypoints.
- Waypoints are generated inside a bounded workspace with a minimum separation.
- Each agent receives a 3‑vector action; the env blends this with the vector to its assigned waypoint, then feeds the resulting position target to a PID controller (from the aviary base).
- Waypoint lifecycle:
  - CLAIMED while an agent is attempting it
  - COMPLETED on success (permanently retired)
  - If an agent stagnates (doesn’t progress), the waypoint is released back to AVAILABLE so it may be reattempted
- Early termination: once all waypoints are COMPLETED and no agents have targets (with a small grace window for visual clarity).


2) Codebase Map
---------------
- env/Marl_dynamic_waypoints.py: Core environment class and logic
- env/BaseRLAviary.py, env/BaseAviary.py: Lightweight aviary scaffolding + PID control interface
- examples/Marl_dyn_train.py.py: Centralized PPO training script, callbacks, plotting
- examples/Marl_dyn_vis.py: GUI visualizer for trained models, overlays, plots
- utils/*: Misc helpers and enums


3) Environment Design (env/Marl_dynamic_waypoints.py)
-----------------------------------------------------
Class: MultiAgentReinforcementLearning
- File reference: env/Marl_dynamic_waypoints.py:28
- Inherits BaseRLAviary (RL‑friendly aviary with PID integration)

Key initialization parameters
- Drone/scene: `num_drones`, `num_waypoints`, `episode_len_sec`, `pyb_freq`, `ctrl_freq`
- Waypoint mechanics: `waypoint_radius`, `waypoint_hold_steps`, `min_waypoint_separation`
- Assignment policy: `priority_mode` in {sequential, distance, random}
- Visualization: `visualize_waypoints`, `waypoint_marker_radius`, `gui`, `record`
- Early termination: `terminate_when_pool_exhausted` (bool), `early_termination_grace_steps` (int steps)

Spaces
- Action space (per agent): 3D Box in [−1, 1]
- Observation space (per agent): base kinematics + waypoint features + neighbor features
  - Base kinematics length is probed dynamically from the BaseRLAviary observation (keeps things robust to base changes)
  - Waypoint features (7): relative position (3), distance (1), has_target flag (1), number of available waypoints (1), completed count so far (1)
  - Neighbor features: for each other agent, relative position (3) + relative velocity (3) → 6×(num_drones−1)

Waypoint generation
- Function: `_generateWaypointPool()` (env/Marl_dynamic_waypoints.py:169)
- Uniform sampling inside `workspace_bounds` until min separation is satisfied or max attempts reached, then fills remaining via uniform sampling

Assignment
- Initial assignment prefers a global solve (Hungarian) if SciPy is available, otherwise falls back to the configured `priority_mode`
- Functions:
  - `_globalAssign()` (env/Marl_dynamic_waypoints.py:203): free agents to available waypoints via a distance cost matrix + Hungarian
  - `_assignInitialWaypoints()` (env/Marl_dynamic_waypoints.py:224): entry point called after reset
  - `_selectWaypointForAgent()` (env/Marl_dynamic_waypoints.py:243): fallback local selection based on priority mode
  - `_claimWaypoint()` (env/Marl_dynamic_waypoints.py:267): state bookkeeping for claim + metrics

Waypoint lifecycle
- `_checkWaypointReached()` (env/Marl_dynamic_waypoints.py:309): within `waypoint_radius` for `waypoint_hold_steps` → success
- `_releaseWaypoint(idx, to_available)` (env/Marl_dynamic_waypoints.py:282):
  - success → COMPLETED (retired)
  - failure (stagnation) → AVAILABLE (retry)
- Waypoint/agent metrics are incremented on claim, success, and failure

Control blending
- Function: `_computeBlendedTarget()` (env/Marl_dynamic_waypoints.py:542)
- Blends two vectors:
  - waypoint_component: unit vector to target × step size (scaled by distance and `ctrl_freq`)
  - agent_component: scaled policy action (less near the target)
- Final target is clipped to workspace bounds and sent to the PID controller in BaseRLAviary

Step loop
- Function: `step(action_dict)` (env/Marl_dynamic_waypoints.py:455)
  - Convert centralized dict action into per‑agent targets with `_computeBlendedTarget`
  - Call super().step to advance physics
  - Stagnation detection: if distance to target doesn’t shrink for several steps → failure path, waypoint returns to AVAILABLE and reassigns
  - Success detection: `_checkWaypointReached` → COMPLETED, assign new waypoint if any
  - Build per‑agent observation dict and reward dict; termination/truncation dicts; info dict

Rewards
- Functions: `_computeRewardDict()` (env/Marl_dynamic_waypoints.py:607), `_computeAgentReward()` (env/Marl_dynamic_waypoints.py:617)
- Components (intuition):
  - Has target bonus
  - Closer distance → higher reward
  - Progress shaping: reward for reducing distance since last step
  - Velocity aligned toward target → positive
  - In‑radius small bonus (promotes stable holding)
  - Stability and altitude penalties
  - Completion bonus (with diminishing returns)
- For verbose debugging: `_computeDetailedReward()` prints component breakdowns

Termination and truncation
- `_computeTerminatedDict()` (env/Marl_dynamic_waypoints.py:795):
  - Per‑agent crash/out‑of‑bounds
  - Global early termination when all waypoints COMPLETED and no agent has a target
  - Grace window: must hold the above for `early_termination_grace_steps` before ending the episode
- `_computeTruncatedDict()` (env/Marl_dynamic_waypoints.py:833): time limit

Info dictionary
- `_computeInfoDict()` (env/Marl_dynamic_waypoints.py:838): for each agent:
  - `waypoints_completed`, `waypoints_assigned`, `waypoints_failed`, `current_target`, `available_waypoints`, `position`
- Episode‑level arrays: `waypoint_counts` with `assigned`, `completed`, `failed` per waypoint index
- Early termination flags: `early_termination`, `early_termination_reason`

Reset and visualization
- `reset()` (env/Marl_dynamic_waypoints.py:871) clears all counters, optionally regenerates the waypoint pool, and assigns initial waypoints
- `_addObstacles()` (env/Marl_dynamic_waypoints.py:912) places visual spheres for waypoints (when GUI on)


4) Training Pipeline (examples/Marl_dyn_train.py.py)
----------------------------------------------------
Centralized PPO
- FlattenDictWrapper flattens the multi‑agent dict into a single Box observation and concatenates per‑agent actions
- VecMonitor -> VecNormalize: vector‑level episode stats + obs/reward normalization
- PPO policy configured with an MLP and standard hyperparameters

Callbacks and logging
- RolloutPrinter: logs recent mean episode reward/length after each rollout
- RewardTrackerCallback: collects episode reward/length/timestep for plotting
- WaypointTrainingCallback: aggregates per‑episode waypoint metrics (assigned/completed/failed), per‑agent completions, and coverage statistics
- EvalCallback: periodically evaluates and writes `best_model.zip`
- Logger: stdout + CSV + TensorBoard (training and waypoint metrics under `waypoints/*`)

Artifacts and plots
- Saved to `--save-path`:
  - `final_model.zip`, `best_model.zip`, `vecnormalize.pkl`
  - Reward plots: `reward_progression.png`, `reward_statistics.png`, raw: `reward_data.npz`
  - Waypoint metrics plots: `waypoint_metrics.png`, raw: `waypoint_metrics.npz`


5) Visualization Pipeline (examples/Marl_dyn_vis.py)
----------------------------------------------------
- Creates a GUI env wrapped in a 1‑env DummyVecEnv; restores `vecnormalize.pkl` if present
- Loads the PPO model (`final_model.zip` or `best_model.zip`)
- Sanity‑checks observation size vs model without resetting the env (avoids double GUI reload)
- Each episode resets with `options={"regenerate_waypoints": True}` to ensure a fresh pool
- Optional slow motion and on‑screen debug overlay:
  - Text above each drone shows: target index, hold counter, and distance
  - Optional green line from drone to its target
- Episode‑level plots at the end: returns, completions/episode, success rate, completions per minute; raw metrics saved as `vis_metrics.npz`


6) Metrics Reference
--------------------
Info dict (per agent)
- `waypoints_completed` (int): completions by this agent this episode
- `waypoints_assigned` (int): assignments to this agent this episode
- `waypoints_failed` (int): stagnation‑based failures for this agent this episode
- `current_target` (int): waypoint index or −1
- `available_waypoints` (int): available waypoint count in pool
- `position` (float[3]): world position

Info dict (episode‑level)
- `waypoint_counts.assigned` (int[num_waypoints]): assigned count per waypoint
- `waypoint_counts.completed` (int[num_waypoints]): completed count per waypoint
- `waypoint_counts.failed` (int[num_waypoints]): failed count per waypoint
- `early_termination` (bool), `early_termination_reason` (str)

Training TensorBoard/CSV scalars
- `train/ep_rew_mean_recent`, `train/ep_len_mean_recent`
- `waypoints/total_assigned`, `waypoints/total_completed`, `waypoints/total_failed`
- `waypoints/mean_completed_per_agent`, `waypoints/std_completed_per_agent`
- `waypoints/success_rate_mean`, `waypoints/coverage_assigned`, `waypoints/coverage_completed`

Saved NPZ structures
- reward_data.npz: `rewards`, `timesteps`, `lengths`
- waypoint_metrics.npz: `timesteps`, `total_*`, `success_rate_mean`, `per_agent_completed`
- vis_metrics.npz: per‑episode arrays for returns, completions, success rate, completions/min, and per‑agent stats


7) CLI Reference (Common Flags)
-------------------------------
Training (examples/Marl_dyn_train.py.py)
- `--timesteps`, `--n-envs`, `--seed`, `--device`
- PPO: `--lr`, `--n-steps`, `--batch-size`, `--n-epochs`, `--gamma`, `--gae-lambda`, `--clip-range`, `--ent-coef`, `--vf-coef`, `--max-grad-norm`
- Env: `--num-drones`, `--num-waypoints`, `--episode-len`, `--waypoint-radius`, `--waypoint-hold-steps`, `--priority-mode`, `--min-waypoint-separation`, `--pyb-freq`, `--ctrl-freq`
- Early termination: `--no-pool-exhaustion-termination`
- Output: `--save-path`, `--log-dir`, `--save-freq`, `--eval-freq`, `--eval-episodes`, `--early-stop-on-reward`, `--eval-deterministic`

Visualizer (examples/Marl_dyn_vis.py)
- `--model`, `--device`, `--episodes`, `--slow-motion`, `--step-delay`, `--pause-between`, `--no-gui-panels`, `--no-shadows`, `--record`, `--out-dir`
- Env params mirror training where relevant: `--num-drones`, `--num-waypoints`, `--episode-len`, `--waypoint-radius`, `--waypoint-hold-steps`, `--priority-mode`, `--min-waypoint-separation`, `--pyb-freq`, `--ctrl-freq`
- Early termination: `--no-pool-exhaustion-termination`
- Debug overlay: `--debug-overlay`, `--highlight-lines`


8) Extension Points
-------------------
Assignment strategies
- Add a new mode in `_selectWaypointForAgent()` and/or modify `_globalAssign()` to incorporate costs (e.g., coverage penalties for frequently visited WPs).

Reward shaping
- Adjust components in `_computeAgentReward()`; keep magnitudes balanced and clip to avoid instability.
  - Common additions: smooth heading alignment rewards, soft collision penalties based on distance thresholds.

Observation design
- Add or remove features in `_getWaypointFeatures()` and `_getNeighborFeatures()`; ensure observation space shape matches.

Additional metrics
- Extend `_computeInfoDict()` and the training callbacks to capture and plot new stats; append to NPZ saves.

Controllers
- The PID controller is supplied by BaseRLAviary; you can switch to velocity control or add new control modes by extending that layer.


9) Troubleshooting & Tips
-------------------------
Obs mismatch (train vs. viz)
- Scripts import the env robustly and check obs dimension; keep both referencing the same `env/Marl_dynamic_waypoints.py`.

Early termination feels “too soon”
- It triggers only when all waypoints are COMPLETED and no agent has a target; a grace window (configurable) keeps the sim briefly alive to visualize the finish.

Waypoint “misses”
- Increase `waypoint_radius` slightly and/or reduce `waypoint_hold_steps`.
- Enable slow motion + overlay to confirm distance/hold counters are behaving as expected.

PyBullet warnings
- Visuals are managed without calling removeBody on reset to avoid C++ warnings.


10) Function‑Level Reference (Environment)
-----------------------------------------
- `__init__` (env/Marl_dynamic_waypoints.py:28): Configure workspace, waypoints, spaces, early termination, and visuals.
- `_log` (env/Marl_dynamic_waypoints.py:164): Conditional print when `verbose` is true.
- `_generateWaypointPool` (env/Marl_dynamic_waypoints.py:169): Build waypoint pool with min separation.
- `_globalAssign` (env/Marl_dynamic_waypoints.py:203): Hungarian assignment for free agents to available waypoints.
- `_assignInitialWaypoints` (env/Marl_dynamic_waypoints.py:224): Seed initial assignments at reset.
- `_selectWaypointForAgent` (env/Marl_dynamic_waypoints.py:243): Local selection per priority mode.
- `_claimWaypoint` (env/Marl_dynamic_waypoints.py:267): Claim bookkeeping and metrics.
- `_releaseWaypoint` (env/Marl_dynamic_waypoints.py:282): Release to AVAILABLE (failure) or mark COMPLETED (success).
- `_checkWaypointReached` (env/Marl_dynamic_waypoints.py:309): Hold‑time success check.
- `_setupMultiAgentSpaces` (env/Marl_dynamic_waypoints.py:350): Build action/observation dict spaces.
- `_computeObs` (env/Marl_dynamic_waypoints.py:380): Per‑agent observation assembly.
- `_getWaypointFeatures` (env/Marl_dynamic_waypoints.py:404): Per‑agent waypoint feature vector.
- `_getNeighborFeatures` (env/Marl_dynamic_waypoints.py:434): Neighbor relative position/velocity features.
- `step` (env/Marl_dynamic_waypoints.py:455): Main loop; blending, stagnation + success checks, reassignment, reward/obs/termination/info.
- `_computeBlendedTarget` (env/Marl_dynamic_waypoints.py:542): Blend action with waypoint vector into a position target.
- `_computeRewardDict` (env/Marl_dynamic_waypoints.py:607): Per‑agent rewards.
- `_computeAgentReward` (env/Marl_dynamic_waypoints.py:617): Reward components & shaping.
- `_computeDetailedReward` (env/Marl_dynamic_waypoints.py:703): Reward with component breakdown (debug).
- `_computeTerminatedDict` (env/Marl_dynamic_waypoints.py:795): Crash and early termination logic.
- `_computeTruncatedDict` (env/Marl_dynamic_waypoints.py:833): Time limit.
- `_computeInfoDict` (env/Marl_dynamic_waypoints.py:838): Per‑agent and per‑episode info.
- `reset` (env/Marl_dynamic_waypoints.py:871): Clear episode state and assign initial waypoints.
- `_addObstacles` (env/Marl_dynamic_waypoints.py:912): Render waypoint visuals in GUI mode.
- `_computeReward` (env/Marl_dynamic_waypoints.py:943): Required by base; sum of per‑agent rewards.
- `_computeTerminated` (env/Marl_dynamic_waypoints.py:948): Any agent terminated.
- `_computeTruncated` (env/Marl_dynamic_waypoints.py:953): Any agent truncated.
- `_computeInfo` (env/Marl_dynamic_waypoints.py:958): Combined info.

