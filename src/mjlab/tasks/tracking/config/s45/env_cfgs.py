"""kuavo S45 flat tracking environment configurations (Corrected for S45 XML)."""

import copy

from mjlab.asset_zoo.robots import get_s45_robot_cfg
from mjlab.actuator import DelayedActuatorCfg
# 注意：你需要确保从对应的 S45 constants 文件中导入 FULL_COLLISION
from mjlab.asset_zoo.robots.kuavo_s45.s45_constants import FULL_COLLISION 
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.tasks.tracking import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise


def kuavo_s45_flat_tracking_env_cfg(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create kuavo S45 flat terrain tracking configuration."""
    cfg = make_tracking_env_cfg()

    robot_cfg = copy.deepcopy(get_s45_robot_cfg())
    robot_cfg.articulation.actuators = tuple(
        DelayedActuatorCfg(
            base_cfg=actuator_cfg,
            delay_target="position",
            delay_min_lag=0,
            delay_max_lag=12,
            delay_hold_prob=0.9,
            delay_update_period=20,
            delay_per_env_phase=True,
        )
        for actuator_cfg in robot_cfg.articulation.actuators
    )
    if play:
        robot_cfg.collisions = (FULL_COLLISION,)
    cfg.scene.entities = {"robot": robot_cfg}

    # ... (传感器配置保持不变) ...


    # ========================================
    # [修正关键点] 修复足部碰撞体正则匹配
    # ========================================
    # S45 的足部碰撞体命名为 left_foot_col1, left_foot_col2 等
    # 原正则 ^(left|right)_foot[1-7]_collision$ 无法匹配，改为：
    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = r"^(left|right)_foot_col[1-7]$"

    # Replace non-physical root-velocity jumps with one-step force pulses.
    cfg.events["push_robot"] = EventTermCfg(
        func=mdp.ScheduledExternalForcePulse,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
            "max_force_n": 150.0,
        },
    )
    cfg.events["base_com"].params["ranges"] = {
        0: (-0.005, 0.005), 1: (-0.005, 0.005), 2: (-0.008, 0.008),
    }
    cfg.events["foot_friction"].params["ranges"] = (0.75, 0.9)
    cfg.events["encoder_bias"].params["bias_range"] = (-0.0025, 0.0025)
    for field in (
        "actuator_gainprm", "actuator_biasprm", "body_mass",
        "dof_frictionloss", "dof_armature",
    ):
        cfg.events[f"expand_{field}"] = EventTermCfg(
            func=mdp.register_domain_randomization_field,
            mode="startup",
            domain_randomization=True,
            params={"field": field},
        )
    cfg.events["pd_gains"] = EventTermCfg(
        func=mdp.randomize_pd_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "kp_range": (0.97, 1.03), "kd_range": (0.97, 1.03),
            "distribution": "uniform", "operation": "scale",
        },
    )
    link_bodies = r"^(leg_[lr][1-6]_link|zarm_[lr][1-7]_link)$"
    cfg.events["base_mass"] = EventTermCfg(
        func=mdp.randomize_field,
        mode="startup",
        domain_randomization=True,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
            "field": "body_mass", "ranges": (0.95, 1.05), "operation": "scale",
        },
    )
    cfg.events["link_mass"] = EventTermCfg(
        func=mdp.randomize_field,
        mode="startup",
        domain_randomization=True,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=link_bodies),
            "field": "body_mass", "ranges": (0.95, 1.05), "operation": "scale",
        },
    )
    cfg.events["link_com"] = EventTermCfg(
        func=mdp.randomize_field,
        mode="startup",
        domain_randomization=True,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=link_bodies),
            "field": "body_ipos",
            "ranges": {0: (-0.01, 0.01), 1: (-0.01, 0.01), 2: (-0.01, 0.01)},
            "operation": "add",
        },
    )
    all_policy_joints = r"^(leg|zarm)_.*_joint$"
    cfg.events["joint_friction"] = EventTermCfg(
        func=mdp.randomize_field,
        mode="startup",
        domain_randomization=True,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=all_policy_joints),
            "field": "dof_frictionloss", "ranges": (0.95, 1.05),
            "operation": "scale",
        },
    )
    cfg.events["joint_armature"] = EventTermCfg(
        func=mdp.randomize_field,
        mode="startup",
        domain_randomization=True,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=all_policy_joints),
            "field": "dof_armature", "ranges": (0.9, 1.1),
            "operation": "scale",
        },
    )

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
    motion_cmd.joint_position_range = (-0.02, 0.02)
    motion_cmd.adaptive_max_probability = 1.0
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
    cfg.terminations["ee_body_pos"].params["ignored_time_range_s"] = (78.0, 82.0)
    cfg.terminations["ee_body_pos"].params["motion_fps"] = 50.0

    hard_segment_params = {
        "hard_segment_s": (78.0, 82.0),
        "hard_scale": 0.6,
        "recovery_s": 2.0,
        "motion_fps": 50.0,
    }
    for reward_name in (
        "motion_global_root_pos", "motion_global_root_ori",
        "motion_body_pos", "motion_body_ori",
        "motion_body_lin_vel", "motion_body_ang_vel",
        "motion_joint_pos", "motion_joint_vel",
    ):
        cfg.rewards[reward_name].params.update(hard_segment_params)
    cfg.rewards["motion_recovery"] = RewardTermCfg(
        func=mdp.motion_recovery_tracking_reward,
        weight=0.5,
        params={
            "command_name": "motion",
            "hard_segment_end_s": 82.0,
            "recovery_s": 2.0,
            "motion_fps": 50.0,
        },
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

    policy_obs = cfg.observations["policy"]
    policy_obs.enable_corruption = True
    final_noise = {
        "motion_target_height": 0.01,
        "motion_anchor_ori_b": 0.025,
        "projected_gravity": 0.025,
        "base_ang_vel": 0.08,
        "joint_pos": 0.01,
        "joint_vel": 0.2,
    }
    for term_name, magnitude in final_noise.items():
        if term_name in policy_obs.terms:
            initial = magnitude * 0.25
            policy_obs.terms[term_name].noise = Unoise(n_min=-initial, n_max=initial)
    for term_name in ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel"):
        term = policy_obs.terms[term_name]
        term.delay_min_lag = 0
        term.delay_max_lag = 3
        term.delay_hold_prob = 0.9
        term.delay_update_period = 5
        term.delay_per_env = True
        term.delay_per_env_phase = True

    cfg.curriculum["real_robot_randomization"] = CurriculumTermCfg(
        func=mdp.RealRobotRandomizationCurriculum(
            start_iteration=17500,
            iterations_per_stage=5000,
            final_stage_iteration=33000,
            rollout_steps=24,
        )
    )
    cfg.curriculum["external_force_ramp"] = CurriculumTermCfg(
        func=mdp.ExternalForceRampCurriculum(
            start_iteration=31500,
            iterations_per_stage=300,
            rollout_steps=24,
        )
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
