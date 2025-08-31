import json
import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from enum import Enum

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

class WaypointStatus(Enum):
    AVAILABLE = 0
    CLAIMED = 1
    COMPLETED = 2

class MultiAgentReinforcementLearning(BaseRLAviary):
    """Multi-agent environment with dynamic waypoint allocation.
    
    Key Features:
    - Shared pool of waypoints that all drones can target
    - Priority-based waypoint claiming system
    - Dynamic waypoint selection after completion
    - Coordination to avoid waypoint conflicts
    """
    
    ENV_VERSION = "0.2.0"
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 num_waypoints: int=20,  # Total number of waypoints in the pool
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int=240,
                 ctrl_freq: int=60, # Lets raise it to 60 from 30 for better control ( it needs to be * of 60)
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.PID,
                 episode_len_sec: int=300,
                 verbose: bool=False,
                 waypoint_radius: float=0.5,
                 waypoint_hold_steps: int=5,
                 priority_mode: str="distance",  # "sequential", "distance", or "random"
                 min_waypoint_separation: float=2.0,  # Minimum distance between waypoints
                 workspace_bounds: tuple=None,  # ((xmin,ymin,zmin), (xmax,ymax,zmax))
                 visualize_waypoints: bool=None,
                 waypoint_marker_radius: float=0.1,
                 # Early termination controls
                 terminate_when_pool_exhausted: bool=True,
                 early_termination_grace_steps: int=1):
        """Initialize environment with dynamic waypoint allocation."""
        
        # Core config
        self.EPISODE_LEN_SEC = episode_len_sec
        self.num_drones = num_drones
        self.num_waypoints = num_waypoints
        self.verbose = verbose
        self.waypoint_radius = waypoint_radius
        self.waypoint_hold_steps = waypoint_hold_steps
        self.priority_mode = priority_mode
        self.min_waypoint_separation = min_waypoint_separation
        self.ctrl_freq = ctrl_freq
        
        # Early termination
        self.terminate_when_pool_exhausted = bool(terminate_when_pool_exhausted)
        self._early_term_reason = None
        # Grace period (in env steps) after the last completion before terminating
        self.early_termination_grace_steps = max(0, int(early_termination_grace_steps))
        self._exhaustion_hold_count = 0
        
        # Workspace bounds for waypoint generation
        if workspace_bounds is None:
            self.workspace_bounds = (
                np.array([-5.0, -5.0, 0.5], dtype=np.float32),
                np.array([5, 5, 2.5], dtype=np.float32)
            )
        else:
            self.workspace_bounds = (
                np.array(workspace_bounds[0], dtype=np.float32),
                np.array(workspace_bounds[1], dtype=np.float32)
            )
        
        # Visualization
        self.visualize_waypoints = bool(gui) if visualize_waypoints is None else bool(visualize_waypoints)
        self.waypoint_marker_radius = float(waypoint_marker_radius)
        
        # Generate shared waypoint pool
        self._generateWaypointPool()
        
        # Initialize waypoint tracking structures
        self.waypoint_status = np.array([WaypointStatus.AVAILABLE] * self.num_waypoints)
        self.waypoint_owner = np.full(self.num_waypoints, -1, dtype=np.int32)  # -1 means no owner
        self.agent_target_waypoint = np.full(self.num_drones, -1, dtype=np.int32)  # -1 means no target
        self.agent_waypoints_completed = {f"agent_{i}": 0 for i in range(num_drones)}
       
        # Additional per-episode metrics
        self.agent_waypoints_assigned = {f"agent_{i}": 0 for i in range(num_drones)}
        self.agent_waypoints_failed = {f"agent_{i}": 0 for i in range(num_drones)}
        self.agent_hold_counter = np.zeros(self.num_drones, dtype=np.int32)
        # Last raw action (removed; can be reintroduced for debugging if needed)
        
        # Track last distance-to-target per agent (for progress shaping)
        self._prev_dist = np.full(self.num_drones, np.inf, dtype=np.float32)

        # Per-waypoint episode counters (for diagnostics & coverage)
        self._wp_assigned_counts = np.zeros(self.num_waypoints, dtype=np.int32)
        self._wp_completed_counts = np.zeros(self.num_waypoints, dtype=np.int32)
        self._wp_failed_counts = np.zeros(self.num_waypoints, dtype=np.int32)
        # Robust total-completed counter (avoids relying on Enum array equality)
        self._completed_total = 0

        # (Removed) Priority queue for assignment; no longer used
        
        # Fixing the idle movements of the drone 
        self._last_dist_to_target = np.full(self.num_drones, np.inf, dtype=np.float32)
        self._stagnant_steps = np.zeros(self.num_drones, dtype=np.int32)

        # Initial positions
        if initial_xyzs is None:
            initial_xyzs = np.zeros((num_drones, 3))
            # Spread drones initially
            for i in range(num_drones):
                angle = 2 * np.pi * i / num_drones
                initial_xyzs[i] = [2 * np.cos(angle), 2 * np.sin(angle), 1.0]
        
        # Parent init
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act
        )
        
        # Setup observation and action spaces
        self._setupMultiAgentSpaces()
        
        # Assign initial waypoints to agents
        self._assignInitialWaypoints()
        
        if self.verbose:
            self._log(f"[INIT] Dynamic waypoint env v{self.ENV_VERSION} agents={num_drones} waypoints={num_waypoints}")
            try:
                info = p.getConnectionInfo(self.CLIENT)
                self._log(f"[PYBULLET] Connection info: {info}")
            except Exception as e:
                self._log(f"[PYBULLET] Could not fetch connection info: {e}")
    
    def _log(self, msg: str):
        """Helper for conditional logging."""
        if self.verbose:
            print(msg)
    
    def _generateWaypointPool(self):
        """Generate a shared pool of waypoints with minimum separation."""
        
        rng = np.random.default_rng()
        bounds_min, bounds_max = self.workspace_bounds
        
        waypoints = []
        attempts = 0
        max_attempts = self.num_waypoints * 100
        
        while len(waypoints) < self.num_waypoints and attempts < max_attempts:
            # Generate random point in workspace
            candidate = bounds_min + rng.random(3) * (bounds_max - bounds_min)
            
            # Check minimum separation from existing waypoints
            valid = True
            for wp in waypoints:
                if np.linalg.norm(candidate - wp) < self.min_waypoint_separation:
                    valid = False
                    break
            
            if valid:
                waypoints.append(candidate)
            attempts += 1
        
        # If we couldn't generate enough with separation, fill remaining randomly
        while len(waypoints) < self.num_waypoints:
            waypoints.append(bounds_min + rng.random(3) * (bounds_max - bounds_min))
        
        self.waypoint_pool = np.array(waypoints, dtype=np.float32)
        
        if self.verbose:
            self._log(f"[WAYPOINTS] Generated {len(self.waypoint_pool)} waypoints")

    def _globalAssign(self):
        """Globally assign free agents to available waypoints using Hungarian.

        Builds a distance cost matrix from each free agent to each available
        waypoint and applies SciPy's linear_sum_assignment to minimize total
        distance. If SciPy is not available, or there are no free agents or
        waypoints, the method returns without changes.
        """
        if not _HAS_SCIPY:
            return
        
        free_agents = [i for i in range(self.num_drones) if self.agent_target_waypoint[i] < 0]
        avail = np.where(self.waypoint_status == WaypointStatus.AVAILABLE)[0]
        
        if len(free_agents) == 0 or len(avail) == 0:
            return
        A = np.zeros((len(free_agents), len(avail)), dtype=np.float32)
        
        for r, i in enumerate(free_agents):
            pos = self._getDroneStateVector(i)[0:3]
            A[r] = np.linalg.norm(self.waypoint_pool[avail] - pos, axis=1)
        rr, cc = linear_sum_assignment(A)
        
        for r, c in zip(rr, cc):
            self._claimWaypoint(free_agents[r], int(avail[c]))

    
    def _assignInitialWaypoints(self):
        """Assign initial waypoints to agents based on priority mode."""
       
        available_indices = np.where(self.waypoint_status == WaypointStatus.AVAILABLE)[0]

        if len(available_indices) == 0:
            self._log("[WARNING] No available waypoints for initial assignment")
            return

        # Use global assignment for initial waypoints to optimize assignment
        if _HAS_SCIPY and len(available_indices) >= self.num_drones:
            self._globalAssign()
        else:
            # Fallback to priority-based assignment
            for i in range(self.num_drones):
                waypoint_idx = self._selectWaypointForAgent(i)
                if waypoint_idx is not None:
                    self._claimWaypoint(i, waypoint_idx)
    
    def _selectWaypointForAgent(self, agent_id: int) -> int:
        """Select best available waypoint for an agent based on priority mode."""
        
        available_indices = np.where(self.waypoint_status == WaypointStatus.AVAILABLE)[0]
        
        if len(available_indices) == 0:
            return None
        
        if self.priority_mode == "random":
            return np.random.choice(available_indices)
        
        elif self.priority_mode == "distance":
            # Choose closest available waypoint
            agent_pos = self._getDroneStateVector(agent_id)[0:3]
            distances = np.linalg.norm(self.waypoint_pool[available_indices] - agent_pos, axis=1)
            return available_indices[np.argmin(distances)]
        
        elif self.priority_mode == "sequential":
            # Take first available waypoint
            return available_indices[0]
        
        else:
            return available_indices[0]
    
    def _claimWaypoint(self, agent_id: int, waypoint_idx: int):
        """Agent claims a waypoint."""
        
        if self.waypoint_status[waypoint_idx] == WaypointStatus.AVAILABLE:
            self.waypoint_status[waypoint_idx] = WaypointStatus.CLAIMED
            self.waypoint_owner[waypoint_idx] = agent_id
            self.agent_target_waypoint[agent_id] = waypoint_idx
            
            # Track assignment attempt
            self.agent_waypoints_assigned[f"agent_{agent_id}"] += 1
            self._wp_assigned_counts[waypoint_idx] += 1
            
            if self.verbose:
                self._log(f"[AGENT {agent_id}] Claimed waypoint {waypoint_idx}")
    
    def _releaseWaypoint(self, waypoint_idx: int, to_available: bool=False):
        """Release a waypoint.

        - If to_available=False: mark as COMPLETED (successful completion)
        - If to_available=True: mark as AVAILABLE (failed attempt)
        """
        if waypoint_idx < 0 or waypoint_idx >= self.num_waypoints:
            return
        
        owner = self.waypoint_owner[waypoint_idx]

        if to_available:
            self.waypoint_status[waypoint_idx] = WaypointStatus.AVAILABLE
            if self.verbose and owner >= 0:
                self._log(f"[WAYPOINT {waypoint_idx}] Released to AVAILABLE (retry)")
        else:
            # Increment total completed once per waypoint
            if self.waypoint_status[waypoint_idx] != WaypointStatus.COMPLETED:
                self._completed_total += 1
            self.waypoint_status[waypoint_idx] = WaypointStatus.COMPLETED
            if self.verbose and owner >= 0:
                self._log(f"[WAYPOINT {waypoint_idx}] Marked COMPLETED")
        
        self.waypoint_owner[waypoint_idx] = -1
        
        # Clear agent's target if this was their waypoint
        if owner >= 0 and owner < self.num_drones:
            if self.agent_target_waypoint[owner] == waypoint_idx:
                self.agent_target_waypoint[owner] = -1
    
    def _checkWaypointReached(self, agent_id: int) -> bool:
        """Check if agent has reached and held at their target waypoint."""
        target_idx = self.agent_target_waypoint[agent_id]
        
        if target_idx < 0 or target_idx >= self.num_waypoints:
            return False
        
        agent_pos = self._getDroneStateVector(agent_id)[0:3]
        target_pos = self.waypoint_pool[target_idx]
        distance = np.linalg.norm(agent_pos - target_pos)
        
        if distance < self.waypoint_radius:
            self.agent_hold_counter[agent_id] += 1
            
            if self.agent_hold_counter[agent_id] >= self.waypoint_hold_steps:
                return True
        else:
            self.agent_hold_counter[agent_id] = 0
        
        return False
    
    # (Removed) _updatePriorityQueue: priority queue logic is unused
    
    def _setupMultiAgentSpaces(self):
        """Define observation and action spaces for multi-agent setup."""

        # Action space: 3D movement for each agent
        self.action_space = spaces.Dict({
            f"agent_{i}": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
            for i in range(self.num_drones)
        })

        # --- Determine base KIN observation length dynamically ---
        # Parent returns per-agent base obs; take one agent and measure its length.
        base_list = super()._computeObs()            # e.g., list/array of length num_drones
        base_dim = int(np.asarray(base_list[0]).ravel().shape[0])
        self._base_obs_dim = base_dim                # store for reference / debugging

        # Waypoint features (7) + neighbor features (rel pos 3 + rel vel 3) per neighbor
        neighbor_dim = 6 * (self.num_drones - 1)
        obs_dim = base_dim + 7 + neighbor_dim

        # Dict observation space: one Box per agent with the computed length
        self.observation_space = spaces.Dict({
            f"agent_{i}": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for i in range(self.num_drones)
        })

        # Optional: helpful log when verbose
        if getattr(self, "verbose", False):
            print(f"[ENV] base_dim={base_dim}, neighbor_dim={neighbor_dim}, "
                f"per_agent_obs={obs_dim}, flattened={obs_dim*self.num_drones}")

    def _computeObs(self):
        """Compute observations for all agents."""
        if self.OBS_TYPE != ObservationType.KIN:
            return {}
        
        all_base = super()._computeObs()  # Get base observations
        obs_dict = {}
        
        for i in range(self.num_drones):
            agent_key = f"agent_{i}"
            base_obs = all_base[i]
            
            # Get waypoint features
            waypoint_features = self._getWaypointFeatures(i)
            
            # Get neighbor features
            neighbor_features = self._getNeighborFeatures(i)
            
            # Combine all features
            full_obs = np.concatenate([base_obs, waypoint_features, neighbor_features])
            obs_dict[agent_key] = full_obs.astype(np.float32)
        
        return obs_dict
    
    def _getWaypointFeatures(self, agent_id: int):
        """Get waypoint-related features for an agent."""
        target_idx = self.agent_target_waypoint[agent_id]
        
        if target_idx >= 0 and target_idx < self.num_waypoints:
            target_pos = self.waypoint_pool[target_idx]
            agent_pos = self._getDroneStateVector(agent_id)[0:3]
            
            # Relative position to target
            rel_pos = target_pos - agent_pos
            distance = np.linalg.norm(rel_pos)
            
            # Number of available waypoints
            num_available = np.sum(self.waypoint_status == WaypointStatus.AVAILABLE)
            
            features = np.array([
                rel_pos[0], rel_pos[1], rel_pos[2],  # Relative position
                distance,  # Distance to target
                1.0,  # Has target
                float(num_available),  # Available waypoints
                float(self.agent_waypoints_completed[f"agent_{agent_id}"])  # Completed count
            ])
        else:
            # No target assigned
            features = np.array([0, 0, 0, 0, 0, 
                                float(np.sum(self.waypoint_status == WaypointStatus.AVAILABLE)),
                                float(self.agent_waypoints_completed[f"agent_{agent_id}"])])
        
        return features
    
    def _getNeighborFeatures(self, agent_id: int):
        """Get features about neighboring agents."""
        my_pos = self._getDroneStateVector(agent_id)[0:3]
        my_vel = self._getDroneStateVector(agent_id)[10:13]
        
        neighbor_features = []
        
        for j in range(self.num_drones):
            if j == agent_id:
                continue
            
            other_pos = self._getDroneStateVector(j)[0:3]
            other_vel = self._getDroneStateVector(j)[10:13]
            
            rel_pos = other_pos - my_pos
            rel_vel = other_vel - my_vel
            
            neighbor_features.extend([*rel_pos, *rel_vel])
        
        return np.array(neighbor_features, dtype=np.float32)
    
    def step(self, action_dict):
        """Advance simulation with waypoint management.

        For PID control, the parent class expects per-agent target positions (length 3).
        Here we blend the learned directional action with the waypoint direction
        to form a next target position for each agent, then pass that to `super().step()`.
        """
        
        full_action = np.zeros((self.num_drones, 3), dtype=np.float32)
        
        for i in range(self.num_drones):
            agent_key = f"agent_{i}"
            raw = np.array(action_dict.get(agent_key, np.zeros(3)), dtype=np.float32)
            raw = np.clip(raw, -1.0, 1.0)
            full_action[i] = self._computeBlendedTarget(i, raw)
            # raw action available here if needed for debugging
        
        # Step physics
        obs_array, reward_float, terminated_bool, truncated_bool, info = super().step(full_action)
        
        for i in range(self.num_drones):
            target_i = self.agent_target_waypoint[i]
            
            if 0 <= target_i < self.num_waypoints:
                d = np.linalg.norm(self._getDroneStateVector(i)[0:3] - self.waypoint_pool[target_i])
            
                if d > self._last_dist_to_target[i] - 1e-3:
                    self._stagnant_steps[i] += 1
                else:
                    self._stagnant_steps[i] = 0
                self._last_dist_to_target[i] = d

                if self._stagnant_steps[i] > 60:  # ~2s @30Hz
                    # Consider this an unsuccessful attempt to complete the waypoint
                    self.agent_waypoints_failed[f"agent_{i}"] += 1
                    if 0 <= target_i < self.num_waypoints:
                        self._wp_failed_counts[target_i] += 1
                    self._releaseWaypoint(target_i, to_available=True)
                    self._globalAssign()
                    self._stagnant_steps[i] = 0


            # Check waypoint reached for each agent and reassign if needed
        for agent in range(self.num_drones):
            
            if self._checkWaypointReached(agent):
                old_target = self.agent_target_waypoint[agent]

                # Update completion count
                agent_key = f"agent_{agent}"
                self.agent_waypoints_completed[agent_key] += 1

                # Mark current waypoint as COMPLETED and optionally reassign globally
                self._releaseWaypoint(old_target, to_available=False)
                if 0 <= old_target < self.num_waypoints:
                    self._wp_completed_counts[old_target] += 1
                self._globalAssign()

                # Reset hold counter
                self.agent_hold_counter[agent] = 0

                # Select and claim new waypoint if still none
                if self.agent_target_waypoint[agent] < 0:
                    new_wp = self._selectWaypointForAgent(agent)
                    if new_wp is not None:
                        self._claimWaypoint(agent, new_wp)

                if self.verbose:
                    self._log(f"[AGENT {agent}] Completed waypoint {old_target}, "
                            f"total: {self.agent_waypoints_completed[agent_key]}")

        # Ensure assignments are refreshed
        self._globalAssign()

        # Ensure any agent without a target immediately gets one if available
        for i in range(self.num_drones):
            if self.agent_target_waypoint[i] < 0:
                wp = self._selectWaypointForAgent(i)
                if wp is not None:
                    self._claimWaypoint(i, wp)

        
        # Compute multi-agent observations and rewards
        obs_dict = self._computeObs()
        reward_dict = self._computeRewardDict()
        terminated_dict = self._computeTerminatedDict()
        truncated_dict = self._computeTruncatedDict()
        info_dict = self._computeInfoDict()
        
        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict
    
    def _computeBlendedTarget(self, agent_id: int, action: np.ndarray) -> np.ndarray:
        """Blend waypoint direction with agent's raw action into a target position.

        This function is the heart of the waypoint-following + RL hybrid approach.
        It takes the agent's raw action (a 3D vector from -1 to 1) and blends it with
        the direction toward the assigned waypoint to create a final target position.

        The blending creates a balance between:
        - Following waypoints reliably (waypoint_component)
        - Allowing RL exploration and adaptation (agent_component)

        Args:
            agent_id: Index of the agent (0 to num_drones-1)
            action: Raw action vector from RL policy, shape (3,), clipped to [-1, 1]

        Returns:
            Target position (x, y, z) for the drone's PID controller
        """
        
        # Get current agent position and target waypoint
        agent_pos = self._getDroneStateVector(agent_id)[0:3]
        target_idx = self.agent_target_waypoint[agent_id]
        action = np.clip(action, -1.0, 1.0)  # Ensure action is within valid range

        if 0 <= target_idx < self.num_waypoints:
            
            # Agent has an assigned waypoint - blend waypoint following with RL action
            wp = self.waypoint_pool[target_idx]
            to_target = wp - agent_pos  # Vector from agent to waypoint
            dist = np.linalg.norm(to_target)  # Distance to waypoint

            if dist > 1 * self.waypoint_radius:
                # Far from waypoint: blend waypoint direction with RL action

                # 1. WAYPOINT COMPONENT: Move directly toward the waypoint
                # Normalization 
                direction = to_target / (dist + 1e-6)  # Unit vector toward waypoint

                # Scale step size based on control frequency for consistent behavior
                # Higher frequency = smaller steps needed for same real-time movement
                base = 0.5 * (self.ctrl_freq / 30.0)  # Base step size (scaled for ctrl_freq)
                step_size = min(base, 0.6 * dist)  # Don't overshoot if waypoint is close
                waypoint_component = direction * step_size  # Movement toward waypoint

                # 2. AGENT COMPONENT: Allow RL policy to influence movement
                # Scale RL influence based on distance - more control when far from waypoint
                # This gives RL more freedom for exploration when not close to target
                agent_scale = 0.15 if dist < 2.0 else 0.25  # Smaller scale when close
                agent_component = action * agent_scale  # RL action contribution

                # 3. FINAL TARGET: Current position + waypoint movement + RL movement
                blended = agent_pos + waypoint_component + agent_component

            else:
                # Close to waypoint: Pin exactly to waypoint for stable holding
                # This ensures the drone reaches and stays at the exact waypoint position
                blended = wp
        else:
            # No assigned waypoint: Allow free movement with some damping
            # RL has full control but movement is scaled down for safety
            blended = agent_pos + action * 0.3

        # Ensure the target position stays within workspace bounds
        # This prevents drones from trying to fly outside the designated area
        blended = np.clip(blended, self.workspace_bounds[0], self.workspace_bounds[1])
        return blended
    
    def _computeRewardDict(self):
        """Compute rewards for all agents."""
        rewards = {}
        
        for i in range(self.num_drones):
            agent_key = f"agent_{i}"
            rewards[agent_key] = self._computeAgentReward(i)
        
        return rewards
    
    def _computeAgentReward(self, agent_id: int):
        """Compute reward for a single agent."""

        reward = 0.0
        components = {} if self.verbose else None

        agent_state = self._getDroneStateVector(agent_id)
        agent_pos = agent_state[0:3]
        agent_vel = agent_state[10:13]

        target_idx = self.agent_target_waypoint[agent_id]

        # Reward for having a target
        if target_idx >= 0:
            reward += 0.5
            if components is not None:
                components['has_target'] = 0.5

            # Distance reward
            target_pos = self.waypoint_pool[target_idx]
            distance = np.linalg.norm(agent_pos - target_pos)
            max_distance = 10.0 
            distance_reward = max(0.0, 1.0 - distance / max_distance)
            reward += distance_reward * 1.0
            if components is not None:
                components['distance'] = distance_reward * 1.0

            # Progress shaping
            prev = self._prev_dist[agent_id]
            if not np.isfinite(prev):
                self._prev_dist[agent_id] = distance
            else:
                progress = prev - distance
                progress_reward = 0.5 * np.clip(progress, -0.5, 0.5)
                reward += progress_reward
                if components is not None:
                    components['progress'] = float(progress_reward)
                self._prev_dist[agent_id] = distance

            # Velocity toward target
            if distance > 0.1:
                direction = (target_pos - agent_pos) / distance
                vel_toward = np.dot(agent_vel, direction)
                vel_reward = vel_toward * 0.2
                reward += vel_reward
                if components is not None:
                    components['velocity'] = float(vel_reward)

            # In-radius bonus
            if distance < self.waypoint_radius:
                reward += 2.0
                if components is not None:
                    components['in_radius'] = 2.0

        else:
            reward -= 0.1
            if components is not None:
                components['no_target'] = -0.1

        # Stability penalties
        roll, pitch = agent_state[7:9]
        if abs(roll) > np.pi/4 or abs(pitch) > np.pi/4:
            reward -= 0.1
            if components is not None:
                components['stability'] = -0.1

        # Altitude penalties
        if agent_pos[2] < 0.5:
            reward -= 0.5
            if components is not None:
                components['altitude_low'] = -0.5
        elif agent_pos[2] > 2.5:
            reward -= 0.1
            if components is not None:
                components['altitude_high'] = -0.1

        # Collision avoidance
        for j in range(self.num_drones):
            if j != agent_id:
                other_pos = self._getDroneStateVector(j)[0:3]
                dist = np.linalg.norm(agent_pos - other_pos)
                if dist < 1.0:
                    penalty = -1.0 * max(0, 1.0 - dist)
                    reward += penalty
                    if components is not None:
                        components[f'collision_{j}'] = float(penalty)

        # Completion bonus with diminishing returns
        if not hasattr(self, '_last_completed_count'):
            self._last_completed_count = {f'agent_{i}': 0 for i in range(self.num_drones)}
        key = f'agent_{agent_id}'
        if key not in self._last_completed_count:
            self._last_completed_count[key] = 0
        current_count = self.agent_waypoints_completed[key]
        if current_count > self._last_completed_count[key]:
            bonus_multiplier = max(0.3, 1.0 - (current_count - 1) * 0.2)
            completion_bonus = 10.0 * bonus_multiplier
            reward += completion_bonus
            if components is not None:
                components['completion'] = float(completion_bonus)
            self._last_completed_count[key] = current_count

        if self.verbose and reward < -1.0:
            self._log(f"[REWARD] Agent {agent_id}: {reward:.2f} - Components: {components}")

        return float(np.clip(reward, -2.0, 8.0))

    # (Removed) _computeDetailedReward: merged into _computeAgentReward when verbose
    
    def _computeTerminatedDict(self):
        """Check termination conditions for all agents, including early termination."""
        terminated = {}
        
        for i in range(self.num_drones):
            agent_key = f"agent_{i}"
            
            state = self._getDroneStateVector(i)
            pos = state[0:3]
            rpy = state[7:10]
            
            # Crash or out of bounds
            if (pos[2] < 0.05 or 
                abs(rpy[0]) > np.pi/2 or 
                abs(rpy[1]) > np.pi/2 or
                np.any(pos < self.workspace_bounds[0] - 5) or
                np.any(pos > self.workspace_bounds[1] + 5)):
                terminated[agent_key] = True
            else:
                terminated[agent_key] = False

        # Early termination: pool exhausted and no active targets
        if self._early_term_reason is None and self.terminate_when_pool_exhausted:
            
            # Robust: use completed counter instead of Enum array compare
            all_completed = (self._completed_total >= self.num_waypoints)
            no_targets = np.all(self.agent_target_waypoint < 0)
            if all_completed and no_targets:
                self._exhaustion_hold_count += 1
                if self._exhaustion_hold_count >= self.early_termination_grace_steps:
                    for i in range(self.num_drones):
                        terminated[f"agent_{i}"] = True
                    self._early_term_reason = "waypoint_pool_exhausted"
            else:
                # Reset hold counter if condition no longer holds
                self._exhaustion_hold_count = 0

        return terminated
    
    def _computeTruncatedDict(self):
        """Check truncation (time limit) for all agents."""
        time_limit_reached = self.step_counter >= int(self.EPISODE_LEN_SEC * self.PYB_FREQ)
        return {f"agent_{i}": time_limit_reached for i in range(self.num_drones)}
    
    def _computeInfoDict(self):
        """Compute info dictionary for all agents, with episode-level diagnostics."""
        info = {}
        
        for i in range(self.num_drones):
            agent_key = f"agent_{i}"
            target_idx = self.agent_target_waypoint[i]
            
            info[agent_key] = {
                "waypoints_completed": self.agent_waypoints_completed[agent_key],
                "waypoints_assigned": self.agent_waypoints_assigned[agent_key],
                "waypoints_failed": self.agent_waypoints_failed[agent_key],
                "current_target": target_idx,
                "available_waypoints": int(np.sum(self.waypoint_status == WaypointStatus.AVAILABLE)),
                "position": self._getDroneStateVector(i)[0:3].copy()
            }

        # Add per-episode waypoint summary (arrays) for diagnostics
        info["waypoint_counts"] = {
            "assigned": self._wp_assigned_counts.copy(),
            "completed": self._wp_completed_counts.copy(),
            "failed": self._wp_failed_counts.copy()
        }

        # Add early termination reason if any
        if self._early_term_reason is not None:
            info["early_termination"] = True
            info["early_termination_reason"] = self._early_term_reason
        else:
            info["early_termination"] = False
        
        return info
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        # Reset waypoint status
        self.waypoint_status[:] = WaypointStatus.AVAILABLE
        self.waypoint_owner[:] = -1
        self.agent_target_waypoint[:] = -1
        self.agent_hold_counter[:] = 0
        self.agent_waypoints_completed = {f"agent_{i}": 0 for i in range(self.num_drones)}
        self.agent_waypoints_assigned = {f"agent_{i}": 0 for i in range(self.num_drones)}
        self.agent_waypoints_failed = {f"agent_{i}": 0 for i in range(self.num_drones)}
        self._wp_assigned_counts[:] = 0
        self._wp_completed_counts[:] = 0
        self._wp_failed_counts[:] = 0
        self._early_term_reason = None
        self._exhaustion_hold_count = 0
        self._completed_total = 0
        
        # Reset progress tracker
        self._prev_dist[:] = np.inf


        # Regenerate waypoint pool if requested
        if options and options.get('regenerate_waypoints', False):
            self._generateWaypointPool()
        
        # No priority queue state to reset
        
        # Parent reset
        obs, info = super().reset(seed=seed, options=options)
        
        # Assign initial waypoints
        self._assignInitialWaypoints()
        
        # Compute observations
        obs_dict = self._computeObs()
        info_dict = self._computeInfoDict()
        
        return obs_dict, info_dict
    
    def _addObstacles(self):
        """Add visualization for waypoints."""
        super()._addObstacles()
        
        if self.GUI and self.visualize_waypoints:
            # Clean up old visuals is unnecessary after reset (bodies are cleared)
            # Create waypoint markers
            for i, waypoint in enumerate(self.waypoint_pool):
                # Color
                color = [0.2, 0.8, 1.0, 0.8]  # pick any RGBA
                
                visual_shape = p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=self.waypoint_marker_radius,
                    rgbaColor=color,
                    physicsClientId=self.CLIENT
                )
                
                body_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=waypoint.tolist(),
                    physicsClientId=self.CLIENT
                )
                # No need to store visual body IDs by default
    
    def _computeReward(self):
        """Required by BaseAviary - returns sum of agent rewards."""
        rewards = self._computeRewardDict()
        return sum(rewards.values())
    
    def _computeTerminated(self):
        """Required by BaseAviary - returns True if any agent terminated."""
        terminated = self._computeTerminatedDict()
        return any(terminated.values())
    
    def _computeTruncated(self):
        """Required by BaseAviary - returns True if time limit reached."""
        truncated = self._computeTruncatedDict()
        return any(truncated.values())
    
    def _computeInfo(self):
        """Required by BaseAviary - returns combined info."""
        return self._computeInfoDict()

# Alias for compatibility
MARLAviary = MultiAgentReinforcementLearning
