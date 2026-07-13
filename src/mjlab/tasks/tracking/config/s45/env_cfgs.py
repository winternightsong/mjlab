"""Kuavo S45 flat motion-tracking configurations."""

from mjlab.asset_zoo.robots import S45_ACTION_SCALE, get_s45_robot_cfg
from mjlab.asset_zoo.robots.kuavo_s45.s45_constants import FULL_COLLISION
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def kuavo_s45_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a S45 tracking config using the mjlab 1.0 observation API."""
  cfg = make_tracking_env_cfg()
  robot_cfg = get_s45_robot_cfg()
  if play:
    robot_cfg.collisions = (FULL_COLLISION,)
  cfg.scene.entities = {"robot": robot_cfg}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = S45_ACTION_SCALE

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
  motion_cmd.anchor_body_name = "base_link"
  motion_cmd.body_names = (
    "base_link",
    "leg_l1_link",
    "leg_l4_link",
    "leg_l6_link",
    "leg_r1_link",
    "leg_r4_link",
    "leg_r6_link",
    "zarm_l1_link",
    "zarm_l4_link",
    "zarm_l7_link",
    "zarm_r1_link",
    "zarm_r4_link",
    "zarm_r7_link",
  )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    r"^left_foot_col[1-7]$",
    r"^right_foot_col[1-7]$",
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  for group_name in ("policy", "critic"):
    group = cfg.observations[group_name]
    if "base_ang_vel" in group.terms:
      group.terms["base_ang_vel"].params["sensor_name"] = "robot/BodyGyro"
    if "base_lin_vel" in group.terms:
      group.terms["base_lin_vel"].params["sensor_name"] = "robot/BodyVel"
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "leg_l6_link",
    "leg_r6_link",
    "zarm_l7_link",
    "zarm_r7_link",
  )
  cfg.viewer.body_name = "base_link"

  if not has_state_estimation:
    actor_terms = {
      name: term
      for name, term in cfg.observations["policy"].terms.items()
      if name not in ("motion_anchor_pos_b", "base_lin_vel")
    }
    cfg.observations["policy"] = ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
