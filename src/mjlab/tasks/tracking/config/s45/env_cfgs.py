"""kuavo S45 flat tracking environment configurations (Corrected for S45 XML)."""

from mjlab.asset_zoo.robots import get_s45_robot_cfg
# 注意：你需要确保从对应的 S45 constants 文件中导入 FULL_COLLISION
from mjlab.asset_zoo.robots.kuavo_s45.s45_constants import FULL_COLLISION 
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def kuavo_s45_flat_tracking_env_cfg(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create kuavo S45 flat terrain tracking configuration."""
    cfg = make_tracking_env_cfg()

    # ... (之前的 robot 实体加载逻辑保持不变) ...
    if play:
        robot_cfg = get_s45_robot_cfg()
        robot_cfg.collisions = (FULL_COLLISION,)
        cfg.scene.entities = {"robot": robot_cfg}
    else:
        cfg.scene.entities = {"robot": get_s45_robot_cfg()}

    # ... (传感器配置保持不变) ...


    # ========================================
    # [修正关键点] 修复足部碰撞体正则匹配
    # ========================================
    # S45 的足部碰撞体命名为 left_foot_col1, left_foot_col2 等
    # 原正则 ^(left|right)_foot[1-7]_collision$ 无法匹配，改为：
    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = r"^(left|right)_foot_col[1-7]$"

    # [修正] 同样检查其他的随机初始化事件（如果存在的话）
    if "reset_robot_offset" in cfg.events:
         cfg.events["reset_robot_offset"].params[
             "asset_cfg"
         ].geom_names = r"^(left|right)_foot_col[1-7]$"

    # ========================================
    # 运动命令与终止条件配置
    # ========================================
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.anchor_body_name = "base_link"
    policy_joint_names = tuple(
        [f"leg_l{i}_joint" for i in range(1, 7)]
        + [f"leg_r{i}_joint" for i in range(1, 7)]
        + [f"zarm_l{i}_joint" for i in range(1, 8)]
        + [f"zarm_r{i}_joint" for i in range(1, 8)]
    )
    motion_cmd.joint_names = policy_joint_names
    # Uncapped late-stage fine-tuning: let failures fully drive start sampling.
    motion_cmd.adaptive_max_probability = 1.0

    # Late-stage fine-tuning: remove joint-space ambiguity left by body-only
    # tracking, while temporarily relaxing action smoothing.
    cfg.rewards["motion_joint_pos"] = RewardTermCfg(
        func=mdp.motion_joint_position_error_exp,
        weight=0.4,
        params={"command_name": "motion", "std": 0.5},
    )
    cfg.rewards["motion_joint_vel"] = RewardTermCfg(
        func=mdp.motion_joint_velocity_error_exp,
        weight=0.15,
        params={"command_name": "motion", "std": 3.0},
    )
    cfg.rewards["action_rate_l2"].weight = -0.05
    
    motion_cmd.body_names = (
        "base_link", "leg_l1_link", "leg_l4_link", "leg_l6_link",
        "leg_r1_link", "leg_r4_link", "leg_r6_link",
        "zarm_l1_link", "zarm_l4_link", "zarm_l7_link",
        "zarm_r1_link", "zarm_r4_link", "zarm_r7_link",
    )

    # CoM 随机化目标
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

    # 终止条件
    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "leg_l6_link", "leg_r6_link", "zarm_l7_link", "zarm_r7_link",
    )

    # ========================================
    # [优化] 显存平衡配置
    # ========================================
    # 如果你要跑 4096 个环境，500 可能导致 24G 显存溢出。
    # 训练初期可以先设为 160，如果训练不稳定再调高。
# ========================================
    # [优化] 仿真与传感器配置
    # ========================================
    cfg.sim.nconmax = 500 

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        fields=("found",),
    )
    cfg.scene.sensors = (self_collision_cfg,)

    # LejuLab Deploy's KUAVO configuration uses a 0.5 position offset scale.
    # Keep residual_action=false there to match use_default_offset=True here.
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.actuator_names = (r"^(leg|zarm)_.*",)
    joint_pos_action.scale = 0.5

    policy_asset_cfg = SceneEntityCfg("robot", joint_names=policy_joint_names)
    for group_name in ("policy", "critic"):
        for term_name in ("joint_pos", "joint_vel"):
            cfg.observations[group_name].terms[term_name].params["asset_cfg"] = (
                policy_asset_cfg
            )

    # 观测名称修正
    cfg.observations["policy"].terms["base_ang_vel"].params["sensor_name"] = "robot/BodyGyro"
    cfg.observations["critic"].terms["base_ang_vel"].params["sensor_name"] = "robot/BodyGyro"
    if "base_lin_vel" in cfg.observations["critic"].terms:
        cfg.observations["critic"].terms["base_lin_vel"].params["sensor_name"] = "robot/BodyVel"

    # ========================================
    # [新增] 修复 Viewer 相机报错
    # ========================================
    # 解决 ValueError: entity_name/body_name required
    cfg.viewer.entity_name = "robot"
    cfg.viewer.body_name = "base_link"

    return cfg
