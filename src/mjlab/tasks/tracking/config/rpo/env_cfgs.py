"""RPO motion-tracking task using the NPZ files shipped by RoboParty."""

from mjlab.asset_zoo.robots import get_rpo_robot_cfg
from mjlab.asset_zoo.robots.rpo.rpo_constants import (
  RPO_BODY_NAMES,
  RPO_MOTION_DIR,
  RPO_ACTION_SCALE,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def rpo_flat_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()
  cfg.scene.entities = {"robot": get_rpo_robot_cfg()}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = RPO_ACTION_SCALE

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = str(RPO_MOTION_DIR / "yundong1.npz")
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = RPO_BODY_NAMES

  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (r".*",)
  for group_name in ("policy", "critic"):
    group = cfg.observations[group_name]
    if "base_ang_vel" in group.terms:
      group.terms["base_ang_vel"].params["sensor_name"] = "robot/angular-velocity"
    if "base_lin_vel" in group.terms:
      group.terms["base_lin_vel"].params["sensor_name"] = "robot/linear-velocity"
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_yaw_link",
    "right_elbow_yaw_link",
  )
  cfg.viewer.body_name = "torso_link"
  cfg.episode_length_s = 20.0

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"
  else:
    # Keep the source training behavior: use the complete state estimate for the
    # critic, but allow the actor's IMU and joint observations to be corrupted.
    policy_terms = dict(cfg.observations["policy"].terms)
    cfg.observations["policy"] = ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  return cfg
