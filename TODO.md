# TODO — Multi-Agent RL Environment

This file tracks planned work, architectural decisions, and change history.


## Recent Changes

### Neural Net Direction + PID Movement Architecture
- **What changed**: The autopilot blending in `_computeBlendedTarget()` has been
  commented out. Previously, the environment automatically blended the neural net
  action with a hardcoded waypoint-direction component. Now the neural network is
  fully responsible for outputting a 3D direction vector, and the PID controllers
  handle the actual movement.
- **Why**: To give the neural network full control over drone direction, making it
  possible to train truly autonomous navigation behavior.
- **Where**: `env/Marl_dynamic_waypoints.py` — `_computeBlendedTarget()` method.
  The old blending code is preserved in comments for reference.

### Per-Drone Neural Agent Architecture
- **What changed**: The architecture now explicitly treats each drone as having its
  own copy of the neural network (shared weights via centralized PPO). Each drone
  receives its own observation (kinematics + waypoint features + neighbor features)
  and produces its own 3D direction action.
- **Where**: Class docstring and documentation updated. The `FlattenDictWrapper`
  slices the flattened action vector so each drone gets its own action segment.

### Documentation Fixes
- Fixed typos: `Marl_dyn_train.py.py` → `Marl_dyn_train.py`
- Updated architecture descriptions in README.md and DETAILED_DESIGN.md
- Removed stale line-number references from function-level documentation
- Removed reference to non-existent `_computeDetailedReward()` function


## Planned Future Work

### Phase 1: Neural Net Direction Control (Current)
- [x] Neural net outputs direction, PID handles movement
- [x] Comment out autopilot blending
- [x] Update documentation to reflect new architecture
- [ ] Train and validate that the neural net can learn to navigate toward waypoints
      using only the observation (waypoint relative position, distance, etc.)
- [ ] Tune reward shaping for the new architecture (the reward already encourages
      moving toward waypoints, but may need adjustment without autopilot)

### Phase 2: Neural Net Motor Control
- [ ] Replace PID controller with direct neural net motor control
- [ ] Neural net outputs RPMs or thrust/torque directly
- [ ] Add motor-level observations (current RPMs, thrust) to the observation space
- [ ] Design new reward components for stable flight without PID

### Phase 3: Independent Per-Drone Networks
- [ ] Consider separate neural network instances per drone (instead of shared weights)
- [ ] Evaluate decentralized training with centralized critic (CTDE)
- [ ] Compare performance: shared weights vs independent networks
- [ ] Explore communication between drone networks

### Phase 4: Advanced Coordination
- [ ] Implement inter-drone communication channels in the observation space
- [ ] Add formation flying objectives
- [ ] Dynamic obstacle avoidance with learned policies
- [ ] Heterogeneous drone types with different capabilities

### Other
- [ ] Add unit tests for the environment
- [ ] Add CI/CD pipeline
- [ ] Package as installable Python module
- [ ] Support for curriculum learning (gradually increase difficulty)
