"""RoboParty RPO constants translated from ``roboparty_train``.

The source project describes the same robot through Isaac Lab's URDF importer.
MjLab uses the accompanying MuJoCo model, while keeping the actuator limits,
gains and initial pose from the training configuration.
"""

import xml.etree.ElementTree as ET

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg, DelayedActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

RPO_ROOT = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "rpo"
RPO_XML = RPO_ROOT / "xmls" / "rpo.xml"
RPO_MOTION_DIR = RPO_ROOT / "motions"


def _load_mjlab_compatible_xml() -> str:
  """Return the source RPO MJCF after MjLab-specific cleanup.

  The source sim2sim MJCF contains motor actuators and actuator telemetry sensors.
  MjLab creates its own native position actuators from the training gains, so those
  source-only elements are removed while the IMU sensors and geometry are kept.
  """
  root = ET.fromstring(RPO_XML.read_text(encoding="utf-8"))
  actuator = root.find("actuator")
  if actuator is not None:
    root.remove(actuator)
  sensors = root.find("sensor")
  if sensors is not None:
    for sensor in list(sensors):
      if sensor.tag.startswith("actuator"):
        sensors.remove(sensor)
  for geom_index, geom in enumerate(root.findall(".//worldbody//geom")):
    if not geom.get("name"):
      geom.set("name", f"rpo_geom_{geom_index}")
  compiler = root.find("compiler")
  if compiler is not None:
    compiler.set("meshdir", (RPO_ROOT / "meshes").as_posix())
  for body_name, site_name in (
    ("left_ankle_roll_link", "left_foot"),
    ("right_ankle_roll_link", "right_foot"),
  ):
    body = root.find(f".//body[@name='{body_name}']")
    if body is not None:
      ET.SubElement(body, "site", name=site_name, pos="0 0 0", size="0.02")
  return ET.tostring(root, encoding="unicode")


def get_spec() -> mujoco.MjSpec:
  """Load a MjLab-compatible RPO MJCF."""
  return mujoco.MjSpec.from_string(_load_mjlab_compatible_xml())


RPO_JOINT_NAMES = (
  "left_thigh_yaw_joint",
  "left_thigh_roll_joint",
  "left_thigh_pitch_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_thigh_yaw_joint",
  "right_thigh_roll_joint",
  "right_thigh_pitch_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "torso_joint",
  "left_arm_pitch_joint",
  "left_arm_roll_joint",
  "left_arm_yaw_joint",
  "left_elbow_pitch_joint",
  "left_elbow_yaw_joint",
  "right_arm_pitch_joint",
  "right_arm_roll_joint",
  "right_arm_yaw_joint",
  "right_elbow_pitch_joint",
  "right_elbow_yaw_joint",
)

RPO_BODY_NAMES = (
  "base_link",
  "left_thigh_yaw_link",
  "left_thigh_roll_link",
  "left_thigh_pitch_link",
  "left_knee_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_thigh_yaw_link",
  "right_thigh_roll_link",
  "right_thigh_pitch_link",
  "right_knee_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_arm_pitch_link",
  "left_arm_roll_link",
  "left_arm_yaw_link",
  "left_elbow_pitch_link",
  "left_elbow_yaw_link",
  "right_arm_pitch_link",
  "right_arm_roll_link",
  "right_arm_yaw_link",
  "right_elbow_pitch_link",
  "right_elbow_yaw_link",
)
RPO_FOOT_SITE_NAMES = ("left_foot", "right_foot")


# Isaac Lab's RPO configuration: 120 Nm/25 rad/s for legs and waist, 27 Nm/8
# rad/s for ankles and arms.  The per-joint gains are retained exactly.
RPO_LEG_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_thigh_yaw_joint",
    ".*_thigh_roll_joint",
    ".*_thigh_pitch_joint",
  ),
  stiffness=100.0,
  damping=3.3,
  effort_limit=120.0,
  armature=0.01,
)
RPO_KNEE_WAIST_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=150.0,
  damping=5.0,
  effort_limit=120.0,
  armature=0.01,
)
RPO_WAIST_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("torso_joint",),
  stiffness=150.0,
  damping=5.0,
  effort_limit=120.0,
  armature=0.01,
)
RPO_FEET_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=40.0,
  damping=2.0,
  effort_limit=27.0,
  armature=0.01,
)
RPO_SHOULDER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_arm_pitch_joint",
    ".*_arm_roll_joint",
    ".*_arm_yaw_joint",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=27.0,
  armature=0.01,
)
RPO_ELBOW_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_elbow_pitch_joint",),
  stiffness=30.0,
  damping=1.5,
  effort_limit=27.0,
  armature=0.01,
)
RPO_ELBOW_YAW_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_elbow_yaw_joint",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=27.0,
  armature=0.01,
)


def _with_training_delay(actuator: BuiltinPositionActuatorCfg) -> DelayedActuatorCfg:
  """Apply RoboParty's 0--2 physics-step actuator delay."""
  return DelayedActuatorCfg(
    base_cfg=actuator,
    delay_min_lag=0,
    delay_max_lag=2,
  )

RPO_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    _with_training_delay(RPO_LEG_ACTUATOR),
    _with_training_delay(RPO_KNEE_WAIST_ACTUATOR),
    _with_training_delay(RPO_WAIST_ACTUATOR),
    _with_training_delay(RPO_FEET_ACTUATOR),
    _with_training_delay(RPO_SHOULDER_ACTUATOR),
    _with_training_delay(RPO_ELBOW_ACTUATOR),
    _with_training_delay(RPO_ELBOW_YAW_ACTUATOR),
  ),
  soft_joint_pos_limit_factor=0.9,
)

RPO_INITIAL_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.75),
  joint_pos={
    "left_thigh_pitch_joint": -0.1,
    "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "left_arm_pitch_joint": 0.18,
    "left_arm_roll_joint": 0.06,
    "left_elbow_pitch_joint": 0.78,
    "right_thigh_pitch_joint": -0.1,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
    "right_arm_pitch_joint": 0.18,
    "right_arm_roll_joint": -0.06,
    "right_elbow_pitch_joint": 0.78,
  },
  joint_vel={".*": 0.0},
)


def get_rpo_robot_cfg() -> EntityCfg:
  """Return a fresh RPO entity configuration."""
  return EntityCfg(
    init_state=RPO_INITIAL_STATE,
    spec_fn=get_spec,
    articulation=RPO_ARTICULATION,
  )


RPO_ACTION_SCALE: dict[str, float] = {
  # Match the RoboParty MuJoCo sim2sim controller, which uses one
  # action-to-position scale for all 23 joints.
  r".*_thigh_yaw_joint": 0.25,
  r".*_thigh_roll_joint": 0.25,
  r".*_thigh_pitch_joint": 0.25,
  r".*_knee_joint": 0.25,
  "torso_joint": 0.25,
  r".*_ankle_pitch_joint": 0.25,
  r".*_ankle_roll_joint": 0.25,
  r".*_arm_pitch_joint": 0.25,
  r".*_arm_roll_joint": 0.25,
  r".*_arm_yaw_joint": 0.25,
  r".*_elbow_pitch_joint": 0.25,
  r".*_elbow_yaw_joint": 0.25,
}
