"""KUAVO S45 velocity environment configurations (Adapted from Kuavo4)."""

import torch

from mjlab.asset_zoo.robots import (
    S45_ACTION_SCALE,
    get_s45_robot_cfg,
)

# 尝试引入 Play 模式下的全碰撞配置 (兼容性写法)
try:
    from mjlab.asset_zoo.robots.kuavo_s45.s45_constants import FULL_COLLISION
    HAS_FULL_COLLISION = True
except ImportError:
    HAS_FULL_COLLISION = False

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


# ---------------------------------------------------------------------------
# 自定义奖励函数：禁止跳跃（强制交替步态）
# ---------------------------------------------------------------------------
def no_hopping_penalty(env, sensor_name: str) -> torch.Tensor:
    """
    惩罚双脚同时腾空（跳跃）。
    """
    contact_found = env.scene[sensor_name].data.found
    num_feet_on_ground = torch.sum(contact_found.float(), dim=1)
    both_feet_in_air = (num_feet_on_ground == 0).float()
    return both_feet_in_air
# ---------------------------------------------------------------------------
# 1. 关节位置与相位全状态跟踪 (Position & Velocity Phase Coupling)
# ---------------------------------------------------------------------------
def reward_gait_arm_phase_coupling(env, arm_pitch_names: list[str], leg_pitch_names: list[str], 
                                   arm_default: float, leg_default: float, gain: float) -> torch.Tensor:
    """
    步态与手臂相位的终极耦合。
    不仅鼓励实际关节位置 (dof_pos) 接近动态生成的参考位置 (ref_dof_pos)，
    同时引入速度相位 (dof_vel) 跟踪，强制手臂的加减速节奏与腿部步态频率完美锁相。
    """
    joint_pos = env.scene["robot"].data.joint_pos
    joint_vel = env.scene["robot"].data.joint_vel
    
    arm_l_idx = env.scene["robot"].find_joints(arm_pitch_names[0])[0]
    arm_r_idx = env.scene["robot"].find_joints(arm_pitch_names[1])[0]
    leg_l_idx = env.scene["robot"].find_joints(leg_pitch_names[0])[0]
    leg_r_idx = env.scene["robot"].find_joints(leg_pitch_names[1])[0]

    # 提取位置与速度并降维
    q_arm_l, q_arm_r = joint_pos[:, arm_l_idx].squeeze(-1), joint_pos[:, arm_r_idx].squeeze(-1)
    q_leg_l, q_leg_r = joint_pos[:, leg_l_idx].squeeze(-1), joint_pos[:, leg_r_idx].squeeze(-1)
    
    v_arm_l, v_arm_r = joint_vel[:, arm_l_idx].squeeze(-1), joint_vel[:, arm_r_idx].squeeze(-1)
    v_leg_l, v_leg_r = joint_vel[:, leg_l_idx].squeeze(-1), joint_vel[:, leg_r_idx].squeeze(-1)

    # --- 动态计算 Reference DOF Pos & Vel ---
    # 左腿配左手(同侧)，因为我们用负号 gain 反转了极性，所以这在物理上等于人类的交叉摆臂
    ref_q_arm_l = arm_default + gain * (q_leg_l - leg_default)
    ref_q_arm_r = arm_default + gain * (q_leg_r - leg_default)
    
    # 速度导数直接映射
    ref_v_arm_l = gain * v_leg_l
    ref_v_arm_r = gain * v_leg_r

    # 计算位置与速度的追踪误差平方
    pos_error = torch.square(q_arm_l - ref_q_arm_l) + torch.square(q_arm_r - ref_q_arm_r)
    vel_error = torch.square(v_arm_l - ref_v_arm_l) + torch.square(v_arm_r - ref_v_arm_r)
    
    # 使用双重高斯核返回归一化奖励 (位置权重极高，速度权重略低作为相位辅助)
    return torch.exp(-pos_error / 0.1) * torch.exp(-vel_error / 10.0)

# ---------------------------------------------------------------------------
# 2. 半周期对称性奖励 (Half Period Symmetry)
# ---------------------------------------------------------------------------
def reward_half_period_symmetry(env, left_arm_names: list[str], right_arm_names: list[str], arm_pitch_default: float) -> torch.Tensor:
    """
    强制左右半周期的关节状态对称。
    无视索引顺序，安全遍历所有手臂关节，严格执行左右交换并取反的逻辑。
    """
    joint_pos = env.scene["robot"].data.joint_pos
    error = torch.zeros(env.num_envs, device=env.device)
    
    # 1. 核心摆臂关节 (Pitch) 的特殊处理 (基于默认姿态的对称)
    l_pitch_idx = env.scene["robot"].find_joints(left_arm_names[0])[0]
    r_pitch_idx = env.scene["robot"].find_joints(right_arm_names[0])[0]
    q_l_pitch = joint_pos[:, l_pitch_idx].squeeze(-1)
    q_r_pitch = joint_pos[:, r_pitch_idx].squeeze(-1)
    
    # q_l - default = -(q_r - default)
    error += torch.square(q_l_pitch + q_r_pitch - 2.0 * arm_pitch_default)

    # 2. 其余所有手臂/手腕关节 (Roll, Yaw) 的直接交换取反 (q_l = -q_r)
    for i in range(1, len(left_arm_names)):
        l_idx = env.scene["robot"].find_joints(left_arm_names[i])[0]
        r_idx = env.scene["robot"].find_joints(right_arm_names[i])[0]
        q_l = joint_pos[:, l_idx].squeeze(-1)
        q_r = joint_pos[:, r_idx].squeeze(-1)
        
        error += torch.square(q_l + q_r)

    return torch.exp(-error / 0.1)
def kuavo_s45_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create kuavo S45 rough terrain velocity configuration."""
    cfg = make_velocity_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 45

    if play and HAS_FULL_COLLISION:
        robot_cfg = get_s45_robot_cfg()
        robot_cfg.collisions = (FULL_COLLISION,)
        cfg.scene.entities = {"robot": robot_cfg}
    else:
        cfg.scene.entities = {"robot": get_s45_robot_cfg()}

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(leg_l6_link|leg_r6_link)$",
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

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = S45_ACTION_SCALE

    cfg.viewer.body_name = "base_link"

    # ========================================
    # [重磅放开] 解除锁定，恢复全向移动！
    # ========================================
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.15
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)  # 允许后退 (-0.5) 到 前进 (1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)  # 允许左右横着走
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)  # 允许左右原地旋转转弯

    # ========================================
    # 传感器名称修正（极度关键！参照Kuavo4）
    # ========================================
    if "base_ang_vel" in cfg.observations["policy"].terms:
        cfg.observations["policy"].terms["base_ang_vel"].params["sensor_name"] = "robot/BodyGyro"
    if "base_lin_vel" in cfg.observations["critic"].terms:
        cfg.observations["critic"].terms["base_lin_vel"].params["sensor_name"] = "robot/BodyVel"

    # ========================================
    # 事件配置
    # ========================================
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = r"^(left|right)_foot_col.*$"
    cfg.events.pop("body_friction", None)

    cfg.events["push_robot"] = EventTermCfg(
        func=envs_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {
                "x": (-1.0, 1.0), 
                "y": (-1.0, 1.0), 
            }
        },
    )

    # ========================================
    # 奖励函数调整
    # ========================================
    
    if "action_rate_l2" in cfg.rewards:
        cfg.rewards["action_rate_l2"].weight = -0.01

    if "track_linear_velocity" in cfg.rewards:
        cfg.rewards["track_linear_velocity"].weight = 3.0
        
    # 既然放开了侧向和自转速度，我们也可以稍微奖励它追踪这些指令 (如果有的话)
    if "track_angular_velocity" in cfg.rewards:
        cfg.rewards["track_angular_velocity"].weight = 1.5

    cfg.rewards["foot_slip"].weight = -1.0
    cfg.rewards["air_time"].weight = 2.0
    cfg.rewards["air_time"].params["threshold_min"] = 0.1

    cfg.rewards["no_hopping"] = RewardTermCfg(
        func=no_hopping_penalty,
        weight=-1.0,  
        params={"sensor_name": "feet_ground_contact"},
    )

# -----------------------------------------------------------------------
    # 注入：步态-手臂相位耦合与半周期对称
    # -----------------------------------------------------------------------
    cfg.rewards["gait_arm_phase_coupling"] = RewardTermCfg(
        func=reward_gait_arm_phase_coupling,
        weight=2.5,  # 给予极高的正向权重，诱导它发现这个完美的步态
        params={
            "arm_pitch_names": ["zarm_l1_joint", "zarm_r1_joint"],
            "leg_pitch_names": ["leg_l3_joint", "leg_r3_joint"],
            "arm_default": 0.0,   
            "leg_default": -0.35, 
            "gain": -1.5  # 核心映射参数
        },
    )

    cfg.rewards["half_period_symmetry"] = RewardTermCfg(
        func=reward_half_period_symmetry,
        weight=1.5,  
        params={
            "left_arm_names": [
                "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", 
                "zarm_l4_joint", "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint"
            ],
            "right_arm_names": [
                "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", 
                "zarm_r4_joint", "zarm_r5_joint", "zarm_r6_joint", "zarm_r7_joint"
            ],
            "arm_pitch_default": 0.0
        },
    )
    # S45 有专门的脚底 site，直接使用 site_names
    site_names = ("l_ft_frame", "r_ft_frame")
    for r_name in ["foot_clearance", "foot_slip", "foot_swing_height"]:
        if r_name in cfg.rewards:
            cfg.rewards[r_name].params["asset_cfg"].site_names = site_names
            cfg.rewards[r_name].params["asset_cfg"].body_names = None

    cfg.rewards.pop("angular_momentum", None)
    cfg.rewards.pop("soft_landing", None)

    cfg.rewards["foot_swing_height"].weight = -5.0 
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # -----------------------------------------------------------------------
    # 姿态约束
    # -----------------------------------------------------------------------
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    
    cfg.rewards["pose"].params["std_walking"] = {
        # 下肢
        r"leg_[lr]3_joint.*": 0.5,   
        r"leg_[lr]1_joint.*": 0.2,   
        r"leg_[lr]2_joint.*": 0.15,  
        r"leg_[lr]4_joint.*": 0.5,   
        r"leg_[lr]5_joint.*": 0.15,  
        r"leg_[lr]6_joint.*": 0.1,   

        # 手臂
        r"zarm_[lr]1_joint.*": 1.0,   # [彻底解放] 让 arm_range 和 arm_symmetry 去管前后摆动
        r"zarm_l2_joint.*": 0.0,    
        r"zarm_r2_joint.*": 0.0,    
        r"zarm_[lr]3_joint.*": 0.0, 
        r"zarm_[lr]4_joint.*": 0.0, 
        r"zarm_[lr]5_joint.*": 0.0, 
        r"zarm_[lr]6_joint.*": 0.0, 
        r"zarm_[lr]7_joint.*": 0.0, 

        # 头部
        r"zhead_[12]_joint.*": 0.0,
    }
    
    cfg.rewards["pose"].params["std_running"] = {
        # 下肢 
        r"leg_[lr]3_joint.*": 0.5,
        r"leg_[lr]1_joint.*": 0.3,     
        r"leg_[lr]2_joint.*": 0.2,
        r"leg_[lr]4_joint.*": 0.5,
        r"leg_[lr]5_joint.*": 0.2,
        r"leg_[lr]6_joint.*": 0.1,

        # 手臂
        r"zarm_[lr]1_joint.*": 1.0,   # [彻底解放]
        r"zarm_l2_joint.*": 0.0,
        r"zarm_r2_joint.*": 0.0,    # (之前为-0.005的隐患已修复)
        r"zarm_[lr]3_joint.*": 0.0,
        r"zarm_[lr]4_joint.*": 0.0,
        r"zarm_[lr]5_joint.*": 0.0,
        r"zarm_[lr]6_joint.*": 0.0,
        r"zarm_[lr]7_joint.*": 0.0,

        # 头部
        r"zhead_[12]_joint.*": 0.0,
    }

    # ========================================
    # Play模式配置
    # ========================================
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )

        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def kuavo_s45_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create kuavo S45 flat terrain velocity configuration."""
    cfg = kuavo_s45_rough_env_cfg(play=play)

    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    # Switch to flat terrain.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Disable terrain curriculum.
    assert "terrain_levels" in cfg.curriculum
    del cfg.curriculum["terrain_levels"]

    if play:
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        # Play 模式同样放开移动限制！
        twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
        twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
        twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

    return cfg