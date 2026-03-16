"""kuavo S45 flat tracking environment configurations (Corrected for S45 XML)."""

from mjlab.asset_zoo.robots import (
    S45_ACTION_SCALE,
    get_s45_robot_cfg,
)
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
    """Create kuavo S45 flat terrain tracking configuration."""
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": get_s45_robot_cfg()}

    # S45 的根节点是 base_link，用于全身碰撞检测
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (self_collision_cfg,)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = S45_ACTION_SCALE

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    # S45 没有独立的 torso_link，base_link 即为躯干
    motion_cmd.anchor_body_name = "base_link"
    
    # 这里的 body_names 用于计算追踪误差，需要对应 S45 的关键骨骼名称
    motion_cmd.body_names = (
        "base_link",             # Root/Pelvis
        "leg_l1_link",           # Left Hip Roll
        "leg_l4_link",           # Left Knee
        "leg_l6_link",           # Left Ankle/Foot
        "leg_r1_link",           # Right Hip Roll
        "leg_r4_link",           # Right Knee
        "leg_r6_link",           # Right Ankle/Foot
        "base_link",             # Torso (Duplicate intended for weighting if needed, or remove)
        "zarm_l2_link",          # Left Shoulder Roll
        "zarm_l4_link",          # Left Elbow
        "zarm_l5_link",          # Left Wrist Yaw (Hand base)
        "zarm_r2_link",          # Right Shoulder Roll
        "zarm_r4_link",          # Right Elbow
        "zarm_r5_link",          # Right Wrist Yaw (Hand base)
    )

    # S45 xml 中足部几何体通常位于 leg_l6_link 下
    # 由于 XML 中可能未给 geom 显式命名，这里尝试匹配 Link 名称相关的 geom
    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = r".*leg_[lr]6_link.*"
    
    # CoM (质心) 随机化目标
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

    # 提前终止条件：检测末端执行器（脚和手）的位置
    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "leg_l6_link",    # Left Foot
        "leg_r6_link",    # Right Foot
        "zarm_l5_link",   # Left Hand
        "zarm_r5_link",   # Right Hand
    )

    cfg.viewer.body_name = "base_link"

    # Modify observations if we don't have state estimation.
    if not has_state_estimation:
        new_policy_terms = {
            k: v
            for k, v in cfg.observations["policy"].terms.items()
            if k not in ["motion_anchor_pos_b", "base_lin_vel"]
        }
        cfg.observations["policy"] = ObservationGroupCfg(
            terms=new_policy_terms,
            concatenate_terms=True,
            enable_corruption=True,
        )

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        # Disable RSI randomization.
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}

        motion_cmd.sampling_mode = "start"

    return cfg