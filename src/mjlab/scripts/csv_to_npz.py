"""Replay motion from CSV for S45 and output to npz with DOF alignment."""

import numpy as np
import torch
import tyro
from tqdm import tqdm
from typing import Any

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
# 确保导入你之前定义的 env_cfg
from mjlab.tasks.tracking.config.s45.env_cfgs import kuavo_s45_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

# 复用你提供的 MotionLoader 类（保持不变）
class MotionLoader:
    # ... (此处省略你代码中已有的 MotionLoader 实现) ...
    def __init__(self, motion_file: str, input_fps: int, output_fps: int, device: torch.device | str, line_range: tuple[int, int] | None = None):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.line_range = line_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        if self.line_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=",", skiprows=self.line_range[0] - 1, max_rows=self.line_range[1] - self.line_range[0] + 1))
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]] # wxyz
        self.motion_dof_poss_input = motion[:, 7:]
        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(self.motion_base_poss_input[index_0], self.motion_base_poss_input[index_1], blend.unsqueeze(1))
        self.motion_base_rots = self._slerp(self.motion_base_rots_input[index_0], self.motion_base_rots_input[index_1], blend)
        self.motion_dof_poss = self._lerp(self.motion_dof_poss_input[index_0], self.motion_dof_poss_input[index_1], blend.unsqueeze(1))

    def _lerp(self, a, b, blend): return a * (1 - blend) + b * blend
    def _slerp(self, a, b, blend):
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]): slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
        return slerped_quats

    def _compute_frame_blend(self, times):
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations, dt):
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self):
        state = (self.motion_base_poss[self.current_idx : self.current_idx + 1], self.motion_base_rots[self.current_idx : self.current_idx + 1], self.motion_base_lin_vels[self.current_idx : self.current_idx + 1], self.motion_base_ang_vels[self.current_idx : self.current_idx + 1], self.motion_dof_poss[self.current_idx : self.current_idx + 1], self.motion_dof_vels[self.current_idx : self.current_idx + 1])
        self.current_idx += 1
        reset_flag = self.current_idx >= self.output_frames
        if reset_flag: self.current_idx = 0
        return state, reset_flag


def run_sim(
    sim: Simulation,
    scene: Scene,
    joint_names,
    input_file,
    input_fps,
    output_fps,
    output_name,
    render,
    line_range,
    renderer: OffscreenRenderer | None = None,
):
    motion = MotionLoader(
        motion_file=input_file,
        input_fps=input_fps,
        output_fps=output_fps,
        device=sim.device,
        line_range=line_range,
    )

    robot: Entity = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    log: dict[str, Any] = {
        "fps": [output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    frames = []
    scene.reset()

    # [关键修正] 获取 CSV 实际的 DOF 数量
    csv_dof_count = motion.motion_dof_poss_input.shape[1]
    train_dof_count = len(joint_names)
    # 计算需要跳过的 DOF (例如 CSV 28个，训练只取 26个，则 skip=2)
    dof_skip = csv_dof_count - train_dof_count 

    pbar = tqdm(total=motion.output_frames, desc="Processing S45 Motion", unit="frame", ncols=100)

    frame_count = 0
    while not file_saved:
        ((motion_base_pos, motion_base_rot, motion_base_lin_vel, motion_base_ang_vel, 
          motion_dof_pos, motion_dof_vel), reset_flag) = motion.get_next_state()

        # 写 Root State
        root_states = robot.data.default_root_state.clone()
        root_states[:, 0:3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # 写 Joint State
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        
        # [关键修复] 根据 skip 逻辑截断 CSV 数据，对齐 S45 的 26 个 DOF
        if dof_skip > 0:
            current_dof_pos = motion_dof_pos[:, :-dof_skip]
            current_dof_vel = motion_dof_vel[:, :-dof_skip]
        else:
            current_dof_pos = motion_dof_pos
            current_dof_vel = motion_dof_vel

        joint_pos[:, robot_joint_indexes] = current_dof_pos
        joint_vel[:, robot_joint_indexes] = current_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.forward()
        scene.update(sim.mj_model.opt.timestep)
        
        if render and renderer is not None:
            renderer.update(sim.data)
            frames.append(renderer.render())

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy())

            frame_count += 1
            pbar.update(1)

            if reset_flag:
                file_saved = True
                pbar.close()
                # ... (此处省略保存 npz 和上传 wandb 的逻辑，保持原样即可) ...
                for k in ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]:
                    log[k] = np.stack(log[k], axis=0)
                np.savez("/tmp/motion.npz", **log)
                # wandb 逻辑等同你提供的代码


def main(
    input_file: str,
    output_name: str,
    input_fps: float = 30.0,
    output_fps: float = 50.0,
    device: str = "cuda:0",
    render: bool = False,
    line_range: tuple[int, int] | None = None,
):
    # ... (初始化 Simulation/Scene/Renderer 的逻辑保持原样) ...
    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / output_fps
    scene = Scene(kuavo_s45_flat_tracking_env_cfg().scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)

    # --- S45 训练关节列表 (26 DOF，不包含头部) ---
    s45_train_joint_names = [
        # 腿部 (12个)
        "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint", "leg_l6_joint",
        "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint", "leg_r6_joint",
        # 手臂 (14个)
        "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint", "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint",
        "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint", "zarm_r5_joint", "zarm_r6_joint", "zarm_r7_joint",
    ]

    run_sim(
        sim=sim,
        scene=scene,
        joint_names=s45_train_joint_names, # 使用 26 DOF 的列表
        input_fps=input_fps,
        input_file=input_file,
        output_fps=output_fps,
        output_name=output_name,
        render=render,
        line_range=line_range,
    )

if __name__ == "__main__":
    tyro.cli(main, config=mjlab.TYRO_FLAGS)