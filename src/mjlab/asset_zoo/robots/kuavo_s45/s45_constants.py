"""Kuavo S45 constants (Strictly adapted from Kuavo4 official configs)."""
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
    ElectricActuator,
    reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

S45_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "robots" / "kuavo_s45" / "xmls" / "s45.xml"
)
# assert S45_XML.exists()

def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, S45_XML.parent / "assets", meshdir)
    return assets

def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(S45_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Actuator config (Adapted from Kuavo4)
##

# Large motor (180 Nm) - 用于大扭矩关节
# 用于：hip roll (leg_l1/r1), knee (leg_l4/r4)
ROTOR_INERTIAS_LARGE = (0.489e-4, 0.109e-4, 0.738e-4)
GEARS_LARGE = (1, 4.5, 5)
ARMATURE_LARGE = reflected_inertia_from_two_stage_planetary(
    ROTOR_INERTIAS_LARGE, GEARS_LARGE
)

# Medium motor (100 Nm) - 用于中等扭矩关节
# 用于：hip yaw/pitch (leg_l2/l3/r2/r3), arm pitch (zarm_l1/r1)
ROTOR_INERTIAS_MEDIUM = (0.489e-4, 0.098e-4, 0.533e-4)
GEARS_MEDIUM = (1, 4.5, 1 + (48 / 22))
ARMATURE_MEDIUM = reflected_inertia_from_two_stage_planetary(
    ROTOR_INERTIAS_MEDIUM, GEARS_MEDIUM
)

# Small motor (50 Nm) - 用于小扭矩关节
# 用于：ankle pitch/roll (leg_l5/r5, leg_l6/r6), arm roll/yaw (zarm_l2/l3/r2/r3), elbow
ROTOR_INERTIAS_SMALL = (0.139e-4, 0.017e-4, 0.169e-4)
GEARS_SMALL = (1, 1 + (46 / 18), 1 + (56 / 16))
ARMATURE_SMALL = reflected_inertia_from_two_stage_planetary(
    ROTOR_INERTIAS_SMALL, GEARS_SMALL
)

# Tiny motor (12 Nm) - 用于微小扭矩关节
# 用于：hand joints (zarm_l5/l6/l7/r5/r6/r7), head (zhead_1/2)
ROTOR_INERTIAS_TINY = (0.068e-4, 0.0, 0.0)
GEARS_TINY = (1, 5, 5)
ARMATURE_TINY = reflected_inertia_from_two_stage_planetary(
    ROTOR_INERTIAS_TINY, GEARS_TINY
)

# Gains calculation
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_LARGE = ARMATURE_LARGE * NATURAL_FREQ**2
DAMPING_LARGE = 2.0 * DAMPING_RATIO * ARMATURE_LARGE * NATURAL_FREQ

STIFFNESS_MEDIUM = ARMATURE_MEDIUM * NATURAL_FREQ**2
DAMPING_MEDIUM = 2.0 * DAMPING_RATIO * ARMATURE_MEDIUM * NATURAL_FREQ

STIFFNESS_SMALL = ARMATURE_SMALL * NATURAL_FREQ**2
DAMPING_SMALL = 2.0 * DAMPING_RATIO * ARMATURE_SMALL * NATURAL_FREQ

STIFFNESS_TINY = ARMATURE_TINY * NATURAL_FREQ**2
DAMPING_TINY = 2.0 * DAMPING_RATIO * ARMATURE_TINY * NATURAL_FREQ


# ========================================
# Joint actuator configs (独立实例化)
# ========================================

# --- 腿部执行器 ---
S45_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(r"leg_[lr]1_joint",),  
    stiffness=STIFFNESS_LARGE,
    damping=DAMPING_LARGE,
    effort_limit=180.0,
    armature=ARMATURE_LARGE,
)

S45_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(r"leg_[lr]2_joint",),
    stiffness=STIFFNESS_MEDIUM,
    damping=DAMPING_MEDIUM,
    effort_limit=100.0,
    armature=ARMATURE_MEDIUM,
)

S45_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(r"leg_[lr]3_joint",),
    stiffness=STIFFNESS_MEDIUM,
    damping=DAMPING_MEDIUM,
    effort_limit=100.0,
    armature=ARMATURE_MEDIUM,
)

S45_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
    target_names_expr=(r"leg_[lr]4_joint",),
    stiffness=STIFFNESS_LARGE,
    damping=DAMPING_LARGE,
    effort_limit=180.0,
    armature=ARMATURE_LARGE,
)

S45_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(r"leg_[lr]5_joint",),
    stiffness=STIFFNESS_SMALL,
    damping=DAMPING_SMALL,
    effort_limit=50.0,
    armature=ARMATURE_SMALL,
)

S45_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(r"leg_[lr]6_joint",),
    stiffness=STIFFNESS_SMALL,
    damping=DAMPING_SMALL,
    effort_limit=50.0,
    armature=ARMATURE_SMALL,
)

# --- 手臂执行器 ---
S45_ACTUATOR_ARM_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(r"zarm_[lr]1_joint",),
    stiffness=STIFFNESS_MEDIUM,
    damping=DAMPING_MEDIUM,
    effort_limit=100.0,
    armature=ARMATURE_MEDIUM,
)

S45_ACTUATOR_ARM_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(r"zarm_[lr]2_joint",),
    stiffness=STIFFNESS_SMALL,
    damping=DAMPING_SMALL,
    effort_limit=50.0,
    armature=ARMATURE_SMALL,
)

S45_ACTUATOR_ARM_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(r"zarm_[lr]3_joint",),
    stiffness=STIFFNESS_SMALL,
    damping=DAMPING_SMALL,
    effort_limit=50.0,
    armature=ARMATURE_SMALL,
)

S45_ACTUATOR_ELBOW = BuiltinPositionActuatorCfg(
    target_names_expr=(r"zarm_[lr]4_joint",),
    stiffness=STIFFNESS_SMALL,
    damping=DAMPING_SMALL,
    effort_limit=50.0,
    armature=ARMATURE_SMALL,
)

S45_ACTUATOR_HAND_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(r"zarm_[lr]5_joint",),
    stiffness=STIFFNESS_TINY,
    damping=DAMPING_TINY,
    effort_limit=12.0,
    armature=ARMATURE_TINY,
)

S45_ACTUATOR_HAND_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(r"zarm_[lr]6_joint",),
    stiffness=STIFFNESS_TINY,
    damping=DAMPING_TINY,
    effort_limit=12.0,
    armature=ARMATURE_TINY,
)

S45_ACTUATOR_HAND_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(r"zarm_[lr]7_joint",),
    stiffness=STIFFNESS_TINY,
    damping=DAMPING_TINY,
    effort_limit=12.0,
    armature=ARMATURE_TINY,
)

# --- 头部执行器 ---
S45_ACTUATOR_HEAD_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(r"zhead_1_joint",),
    stiffness=STIFFNESS_TINY,
    damping=DAMPING_TINY,
    effort_limit=1.5,
    armature=ARMATURE_TINY,
)

S45_ACTUATOR_HEAD_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(r"zhead_2_joint",),
    stiffness=STIFFNESS_TINY,
    damping=DAMPING_TINY,
    effort_limit=12.0, # 匹配 Tiny 限制
    armature=ARMATURE_TINY,
)

# ========================================
# Keyframe config (HOME_KEYFRAME)
# ========================================

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.9),  # 站立高度
    joint_pos={
        # 腿部 - 直立姿态
        r"leg_[lr]1_joint": 0.0,      # Hip roll
        r"leg_[lr]2_joint": 0.0,      # Hip yaw
        r"leg_[lr]3_joint": -0.1,     # Hip pitch
        r"leg_[lr]4_joint": 0.3,      # Knee
        r"leg_[lr]5_joint": -0.15,    # Ankle pitch
        r"leg_[lr]6_joint": 0.0,      # Ankle roll

        # 手臂 - 自然下垂姿态
        r"zarm_[lr]1_joint": 0.0,     # Arm pitch0.3
        r"zarm_l2_joint": 0.2,        # Left arm roll
        r"zarm_r2_joint": -0.2,       # Right arm roll (负值！)
        r"zarm_[lr]3_joint": 0.0,     # Arm yaw
        r"zarm_[lr]4_joint": -0.5,    # Elbow
        r"zarm_[lr]5_joint": 0.0,     # Hand yaw
        r"zarm_l6_joint": 0.0,        # Left wrist roll
        r"zarm_r6_joint": 0.0,        # Right wrist roll
        r"zarm_[lr]7_joint": 0.0,     # Hand pitch

        # 头部 - 中性位置
        r"zhead_[12]_joint": 0.0,     # Head joints
    },
    joint_vel={".*": 0.0},
)

# ========================================
# Collision config
# ========================================
# 为了兼容 S45 的 XML 命名，将 Kuavo4 的脚部碰撞正则稍微放宽适应 S45
FOOT_COLLISION_REGEX = r"^(left|right)_foot_col.*$"

# 训练模式：只启用脚与地面碰撞（提高效率）
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
    geom_names_expr=(r".*",),          # S45 中可能没有 _collision 后缀
    contype=0,                         # 禁用自身接触
    conaffinity=1,                     # 只与terrain接触
    condim={
        FOOT_COLLISION_REGEX: 3,       # 脚部3D摩擦
        r".*": 1,                      # 其他1D摩擦
    },
    priority={
        FOOT_COLLISION_REGEX: 1,       # 脚部高优先级
    },
    friction={
        FOOT_COLLISION_REGEX: (0.6,),  # 脚部摩擦系数
    },
)

# Play模式：启用所有碰撞（包括自碰撞）
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(r".*",),
    condim={
        FOOT_COLLISION_REGEX: 3,  
        r".*": 1,                 
    },
    priority={
        FOOT_COLLISION_REGEX: 1,
    },
    friction={
        FOOT_COLLISION_REGEX: (0.6,),
    },
)

# ========================================
# Final config
# ========================================

S45_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        S45_ACTUATOR_HIP_ROLL,
        S45_ACTUATOR_HIP_YAW,
        S45_ACTUATOR_HIP_PITCH,
        S45_ACTUATOR_KNEE,
        S45_ACTUATOR_ANKLE_PITCH,
        S45_ACTUATOR_ANKLE_ROLL,
        S45_ACTUATOR_ARM_PITCH,
        S45_ACTUATOR_ARM_ROLL,
        S45_ACTUATOR_ARM_YAW,
        S45_ACTUATOR_ELBOW,
        S45_ACTUATOR_HAND_YAW,
        S45_ACTUATOR_HAND_ROLL,
        S45_ACTUATOR_HAND_PITCH,
        S45_ACTUATOR_HEAD_YAW,
        S45_ACTUATOR_HEAD_PITCH,
    ),
    soft_joint_pos_limit_factor=0.9,
)

def get_s45_robot_cfg() -> EntityCfg:
    """获取S45机器人配置实例
    每次调用返回新实例，避免配置共享导致的变异问题。
    """
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FULL_COLLISION_WITHOUT_SELF,),  # 默认训练用
        spec_fn=get_spec,
        articulation=S45_ARTICULATION,
    )

# 计算 action scale (完全按照官方逻辑还原)
S45_ACTION_SCALE: dict[str, float] = {}
for a in S45_ARTICULATION.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    e = a.effort_limit
    s = a.stiffness
    names = a.target_names_expr
    assert e is not None
    for n in names:
        S45_ACTION_SCALE[n] = 0.25 * e / s

if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_s45_robot_cfg())
    viewer.launch(robot.spec.compile())