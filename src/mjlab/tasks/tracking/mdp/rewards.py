from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def _hard_segment_imitation_scale(
  command: MotionCommand,
  hard_segment_s: tuple[float, float] | None,
  hard_scale: float,
  recovery_s: float,
  motion_fps: float,
) -> torch.Tensor:
  """Return a per-environment imitation scale with smooth post-segment recovery."""
  scale = torch.ones(command.num_envs, device=command.device)
  if hard_segment_s is None:
    return scale
  time_s = command.time_steps.float() / motion_fps
  start_s, end_s = hard_segment_s
  in_segment = (time_s >= start_s) & (time_s <= end_s)
  scale[in_segment] = hard_scale
  if recovery_s > 0.0:
    in_recovery = (time_s > end_s) & (time_s < end_s + recovery_s)
    progress = (time_s[in_recovery] - end_s) / recovery_s
    scale[in_recovery] = hard_scale + (1.0 - hard_scale) * progress
  return scale


def _scale_imitation_reward(
  reward: torch.Tensor,
  command: MotionCommand,
  hard_segment_s: tuple[float, float] | None,
  hard_scale: float,
  recovery_s: float,
  motion_fps: float,
) -> torch.Tensor:
  return reward * _hard_segment_imitation_scale(
    command, hard_segment_s, hard_scale, recovery_s, motion_fps
  )


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float,
  hard_segment_s: tuple[float, float] | None = None, hard_scale: float = 1.0,
  recovery_s: float = 0.0, motion_fps: float = 50.0,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return _scale_imitation_reward(torch.exp(-error / std**2), command, hard_segment_s, hard_scale, recovery_s, motion_fps)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float,
  hard_segment_s: tuple[float, float] | None = None, hard_scale: float = 1.0,
  recovery_s: float = 0.0, motion_fps: float = 50.0,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return _scale_imitation_reward(torch.exp(-error / std**2), command, hard_segment_s, hard_scale, recovery_s, motion_fps)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
  hard_segment_s: tuple[float, float] | None = None, hard_scale: float = 1.0,
  recovery_s: float = 0.0, motion_fps: float = 50.0,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return _scale_imitation_reward(torch.exp(-error.mean(-1) / std**2), command, hard_segment_s, hard_scale, recovery_s, motion_fps)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
  hard_segment_s: tuple[float, float] | None = None, hard_scale: float = 1.0,
  recovery_s: float = 0.0, motion_fps: float = 50.0,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return _scale_imitation_reward(torch.exp(-error.mean(-1) / std**2), command, hard_segment_s, hard_scale, recovery_s, motion_fps)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
  hard_segment_s: tuple[float, float] | None = None, hard_scale: float = 1.0,
  recovery_s: float = 0.0, motion_fps: float = 50.0,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return _scale_imitation_reward(torch.exp(-error.mean(-1) / std**2), command, hard_segment_s, hard_scale, recovery_s, motion_fps)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
  hard_segment_s: tuple[float, float] | None = None, hard_scale: float = 1.0,
  recovery_s: float = 0.0, motion_fps: float = 50.0,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return _scale_imitation_reward(torch.exp(-error.mean(-1) / std**2), command, hard_segment_s, hard_scale, recovery_s, motion_fps)


def motion_joint_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float,
  hard_segment_s: tuple[float, float] | None = None, hard_scale: float = 1.0,
  recovery_s: float = 0.0, motion_fps: float = 50.0,
) -> torch.Tensor:
  """Reward joint-position tracking in the policy joint order."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.mean(torch.square(command.joint_pos - command.robot_joint_pos), dim=-1)
  return _scale_imitation_reward(torch.exp(-error / std**2), command, hard_segment_s, hard_scale, recovery_s, motion_fps)


def motion_joint_velocity_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float,
  hard_segment_s: tuple[float, float] | None = None, hard_scale: float = 1.0,
  recovery_s: float = 0.0, motion_fps: float = 50.0,
) -> torch.Tensor:
  """Reward joint-velocity tracking in the policy joint order."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.mean(torch.square(command.joint_vel - command.robot_joint_vel), dim=-1)
  return _scale_imitation_reward(torch.exp(-error / std**2), command, hard_segment_s, hard_scale, recovery_s, motion_fps)


def motion_recovery_tracking_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  hard_segment_end_s: float,
  recovery_s: float = 2.0,
  motion_fps: float = 50.0,
  joint_std: float = 0.5,
  body_std: float = 0.3,
) -> torch.Tensor:
  """Reward safely rejoining the reference during the post-hard-segment window."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  time_s = command.time_steps.float() / motion_fps
  active = (time_s > hard_segment_end_s) & (time_s < hard_segment_end_s + recovery_s)
  joint_error = torch.mean(torch.square(command.joint_pos - command.robot_joint_pos), dim=-1)
  body_error = torch.mean(torch.sum(torch.square(command.body_pos_relative_w - command.robot_body_pos_w), dim=-1), dim=-1)
  quality = 0.5 * torch.exp(-joint_error / joint_std**2) + 0.5 * torch.exp(-body_error / body_std**2)
  progress = ((time_s - hard_segment_end_s) / recovery_s).clamp(0.0, 1.0)
  return active.float() * quality * (0.5 + 0.5 * progress)


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Cost that returns the number of self-collisions detected by a sensor."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)
