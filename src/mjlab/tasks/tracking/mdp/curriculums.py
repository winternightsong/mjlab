from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class RealRobotRandomizationCurriculum:
  """Increase S45 sim-to-real randomization every fixed PPO iteration block."""

  def __init__(
    self,
    start_iteration: int,
    iterations_per_stage: int,
    rollout_steps: int,
    final_stage_iteration: int | None = None,
  ):
    self.start_iteration = start_iteration
    self.iterations_per_stage = iterations_per_stage
    self.rollout_steps = rollout_steps
    self.final_stage_iteration = final_stage_iteration
    self._stage = -1

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> dict[str, float]:
    del env_ids
    iteration = env.common_step_counter // self.rollout_steps
    stage = max(0, min(3, (iteration - self.start_iteration) // self.iterations_per_stage))
    if self.final_stage_iteration is not None and iteration < self.final_stage_iteration:
      stage = min(stage, 2)
    if stage != self._stage:
      self._apply_stage(env, stage)
      self._stage = stage
    return {"stage": float(stage), "iteration": float(iteration)}

  def _apply_stage(self, env: ManagerBasedRlEnv, stage: int) -> None:
    scales = (0.25, 0.50, 0.75, 1.0)
    scale = scales[stage]

    com_xy = (0.005, 0.01, 0.015, 0.02)[stage]
    com_z = (0.008, 0.015, 0.023, 0.03)[stage]
    com_term = env.event_manager.get_term_cfg("base_com")
    com_term.params["ranges"] = {
      0: (-com_xy, com_xy), 1: (-com_xy, com_xy), 2: (-com_z, com_z)
    }
    com_term.func(env, None, **com_term.params)

    friction = ((0.75, 0.9), (0.62, 0.95), (0.52, 1.0), (0.45, 1.0))[stage]
    friction_term = env.event_manager.get_term_cfg("foot_friction")
    friction_term.params["ranges"] = friction
    friction_term.func(env, None, **friction_term.params)

    base_mass = ((0.95, 1.05), (0.9, 1.15), (0.85, 1.3), (0.8, 1.5))[stage]
    link_mass = ((0.95, 1.05), (0.9, 1.1), (0.85, 1.15), (0.8, 1.2))[stage]
    for term_name, ranges in (("base_mass", base_mass), ("link_mass", link_mass)):
      term = env.event_manager.get_term_cfg(term_name)
      term.params["ranges"] = ranges
      term.func(env, None, **term.params)

    link_com = (0.01, 0.02, 0.03, 0.04)[stage]
    link_com_term = env.event_manager.get_term_cfg("link_com")
    link_com_term.params["ranges"] = {
      0: (-link_com, link_com), 1: (-link_com, link_com),
      2: (-link_com, link_com),
    }
    link_com_term.func(env, None, **link_com_term.params)

    joint_friction = ((0.95, 1.05), (0.9, 1.1), (0.85, 1.15), (0.8, 1.2))[stage]
    joint_armature = ((0.9, 1.1), (0.75, 1.25), (0.6, 1.4), (0.5, 1.5))[stage]
    for term_name, ranges in (
      ("joint_friction", joint_friction), ("joint_armature", joint_armature)
    ):
      term = env.event_manager.get_term_cfg(term_name)
      term.params["ranges"] = ranges
      term.func(env, None, **term.params)

    gain = (0.03, 0.07, 0.11, 0.15)[stage]
    gain_term = env.event_manager.get_term_cfg("pd_gains")
    gain_term.params["kp_range"] = (1.0 - gain, 1.0 + gain)
    gain_term.params["kd_range"] = (1.0 - gain, 1.0 + gain)
    gain_term.func(env, None, **gain_term.params)

    encoder = env.event_manager.get_term_cfg("encoder_bias")
    encoder.params["bias_range"] = (-0.01 * scale, 0.01 * scale)
    encoder.func(env, None, **encoder.params)

    joint = (0.02, 0.03, 0.04, 0.05)[stage]
    motion_term = env.command_manager.get_term("motion")
    motion_term.cfg.joint_position_range = (-joint, joint)

    motor_ranges = ((0, 2), (2, 5), (5, 9), (8, 12))
    motor_min, motor_max = motor_ranges[stage]
    robot = env.scene["robot"]
    for actuator in robot.actuators:
      for buffer in getattr(actuator, "_delay_buffers", {}).values():
        buffer.min_lag, buffer.max_lag = motor_min, motor_max
        lags = torch.randint(motor_min, motor_max + 1, (env.num_envs,), device=env.device)
        buffer.set_lags(lags)

    observation_lags = ((0, 0), (0, 1), (1, 2), (2, 3))
    obs_min, obs_max = observation_lags[stage]
    manager = env.observation_manager
    for term_name in ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel"):
      buffer = manager._group_obs_term_delay_buffer["policy"][term_name]
      buffer.min_lag, buffer.max_lag = obs_min, obs_max
      buffer.set_lags(torch.randint(obs_min, obs_max + 1, (env.num_envs,), device=env.device))

    final_noise = {
      "motion_target_height": 0.01, "motion_anchor_ori_b": 0.025,
      "projected_gravity": 0.025, "base_ang_vel": 0.08,
      "joint_pos": 0.01, "joint_vel": 0.2,
    }
    for term_name, magnitude in final_noise.items():
      noise = manager.get_term_cfg("policy", term_name).noise
      if noise is not None:
        noise.n_min, noise.n_max = -magnitude * scale, magnitude * scale
        noise._tensor_cache.clear()


class ExternalForceRampCurriculum:
  """Ramp only the new force pulse while the main randomization stays at stage 2/3."""

  def __init__(self, start_iteration: int, iterations_per_stage: int, rollout_steps: int):
    self.start_iteration = start_iteration
    self.iterations_per_stage = iterations_per_stage
    self.rollout_steps = rollout_steps
    self._stage = -1

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> dict[str, float]:
    del env_ids
    iteration = env.common_step_counter // self.rollout_steps
    stage = max(0, min(3, (iteration - self.start_iteration) // self.iterations_per_stage))
    if stage != self._stage:
      xy_force = (150.0, 300.0, 600.0, 900.0)[stage]
      force_term = env.event_manager.get_term_cfg("push_robot")
      force_term.params["max_force_n"] = xy_force
      self._stage = stage
    return {
      "stage": float(stage),
      "iteration": float(iteration),
      "xy_force_max_n": (150.0, 300.0, 600.0, 900.0)[stage],
    }
