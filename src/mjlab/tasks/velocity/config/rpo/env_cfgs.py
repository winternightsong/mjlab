"""RPO velocity tasks using the MjLab 1.0 manager API."""

import torch

from mjlab.asset_zoo.robots import get_rpo_robot_cfg
from mjlab.asset_zoo.robots.rpo.rpo_constants import RPO_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.viewer import ViewerConfig
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def rpo_linear_velocity_error_l2(env, command_name: str) -> torch.Tensor:
  """Directly penalize residual xy velocity error."""
  robot = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  error = command[:, :2] - robot.data.root_link_lin_vel_b[:, :2]
  return torch.sum(torch.square(error), dim=1)


def rpo_body_distance_range_penalty(
  env,
  asset_cfg,
  min_distance: float,
  max_distance: float,
) -> torch.Tensor:
  """Penalize pairwise lateral distances outside a valid walking range."""
  robot = env.scene["robot"]
  positions_y = robot.data.body_link_pos_w[:, asset_cfg.body_ids, 1]
  distance = torch.abs(positions_y[:, 0] - positions_y[:, 1])
  below = torch.relu(min_distance - distance) / min_distance
  above = torch.relu(distance - max_distance) / max_distance
  return torch.square(below) + torch.square(above)


def rpo_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {"robot": get_rpo_robot_cfg()}
  feet = ("left_foot", "right_foot")

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = RPO_ACTION_SCALE
  cfg.viewer.body_name = "torso_link"
  for group_name in ("policy", "critic"):
    group = cfg.observations[group_name]
    if "base_ang_vel" in group.terms:
      group.terms["base_ang_vel"].params["sensor_name"] = "robot/angular-velocity"
    if "base_lin_vel" in group.terms:
      group.terms["base_lin_vel"].params["sensor_name"] = "robot/linear-velocity"
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.viz.z_offset = 1.15
  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = feet
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (r".*",)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].weight = -0.05
  for name in ("foot_clearance", "foot_swing_height", "foot_slip"):
    cfg.rewards[name].params["asset_cfg"].site_names = feet

  cfg.rewards["pose"].params["std_standing"] = {
    r".*_thigh_(yaw|roll|pitch)_joint": 0.05,
    r".*_knee_joint": 0.08,
    r".*_ankle_(pitch|roll)_joint": 0.05,
    r"torso_joint": 0.05,
    r".*_arm_(pitch|roll|yaw)_joint": 0.08,
    r".*_elbow_(pitch|yaw)_joint": 0.08,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_thigh_yaw_joint": 0.15,
    r".*_thigh_roll_joint": 0.2,
    r".*_thigh_pitch_joint": 0.5,
    r".*_knee_joint": 0.5,
    r".*_ankle_pitch_joint": 0.2,
    r".*_ankle_roll_joint": 0.12,
    r"torso_joint": 0.2,
    r".*_arm_pitch_joint": 0.8,
    r".*_arm_roll_joint": 0.2,
    r".*_arm_yaw_joint": 0.15,
    r".*_elbow_pitch_joint": 0.3,
    r".*_elbow_yaw_joint": 0.15,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*_thigh_yaw_joint": 0.2,
    r".*_thigh_roll_joint": 0.3,
    r".*_thigh_pitch_joint": 0.6,
    r".*_knee_joint": 0.7,
    r".*_ankle_pitch_joint": 0.3,
    r".*_ankle_roll_joint": 0.15,
    r"torso_joint": 0.3,
    r".*_arm_pitch_joint": 1.0,
    r".*_arm_roll_joint": 0.3,
    r".*_arm_yaw_joint": 0.2,
    r".*_elbow_pitch_joint": 0.4,
    r".*_elbow_yaw_joint": 0.2,
  }

  cfg.rewards["linear_velocity_error_l2"] = RewardTermCfg(
    func=rpo_linear_velocity_error_l2,
    weight=-1.0,
    params={"command_name": "twist"},
  )
  cfg.rewards["feet_distance"] = RewardTermCfg(
    func=rpo_body_distance_range_penalty,
    weight=-0.2,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link")
      ),
      "min_distance": 0.16,
      "max_distance": 0.50,
    },
  )
  cfg.rewards["knee_distance"] = RewardTermCfg(
    func=rpo_body_distance_range_penalty,
    weight=-0.2,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=("left_knee_link", "right_knee_link")
      ),
      "min_distance": 0.18,
      "max_distance": 0.35,
    },
  )

  cfg.rewards["air_time"].weight = 0.0
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name},
  )

  if play:
    # Keep the play camera fixed in world coordinates.  Tracking torso_link
    # makes small residual body motions look like the ground is shaking.
    cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
    cfg.viewer.entity_name = None
    cfg.viewer.body_name = None
    cfg.viewer.lookat = (0.0, 0.0, 0.5)
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0
  return cfg


def rpo_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = rpo_rough_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.observations["critic"].terms.pop("foot_height", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)
  if play:
    command = cfg.commands["twist"]
    assert isinstance(command, UniformVelocityCommandCfg)
    command.ranges.lin_vel_x = (-1.0, 1.0)
    command.ranges.lin_vel_y = (-0.5, 0.5)
    command.ranges.ang_vel_z = (-0.7, 0.7)
  return cfg
