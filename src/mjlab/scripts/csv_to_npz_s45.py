"""Convert a Kuavo S45 CSV motion with the mjlab 1.0 converter."""

import os
import tempfile

import numpy as np
import torch
import tyro

import mjlab
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import run_sim
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.s45.env_cfgs import kuavo_s45_flat_tracking_env_cfg
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


S45_MOTION_JOINTS = (
  "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint",
  "leg_l5_joint", "leg_l6_joint", "leg_r1_joint", "leg_r2_joint",
  "leg_r3_joint", "leg_r4_joint", "leg_r5_joint", "leg_r6_joint",
  "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint",
  "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint", "zarm_r1_joint",
  "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint", "zarm_r5_joint",
  "zarm_r6_joint", "zarm_r7_joint",
)


def main(
  input_file: str,
  output_name: str,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  line_range: tuple[int, int] | None = None,
):
  """Replay S45 CSV data and save a standard mjlab motion NPZ."""
  if device.startswith("cuda") and not torch.cuda.is_available():
    device = "cpu"
  # Older S45 CSV exports contain two extra head joints.  Keep the documented
  # 26-DOF training order without changing the generic converter used by G1.
  if line_range is None:
    motion = np.loadtxt(input_file, delimiter=",")
  else:
    motion = np.loadtxt(
      input_file,
      delimiter=",",
      skiprows=line_range[0] - 1,
      max_rows=line_range[1] - line_range[0] + 1,
    )
  dof_count = motion.shape[1] - 7
  temp_path = None
  if dof_count > len(S45_MOTION_JOINTS):
    fd, temp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    np.savetxt(temp_path, motion[:, : 7 + len(S45_MOTION_JOINTS)], delimiter=",")
    input_file = temp_path
    line_range = None

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  scene = Scene(kuavo_s45_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  renderer = None
  if render:
    renderer = OffscreenRenderer(
      model=sim.mj_model,
      cfg=ViewerConfig(
        height=480,
        width=640,
        origin_type=ViewerConfig.OriginType.ASSET_ROOT,
        distance=2.0,
        elevation=-5.0,
        azimuth=20,
      ),
      scene=scene,
    )
    renderer.initialize()
  try:
    run_sim(
      sim=sim,
      scene=scene,
      joint_names=S45_MOTION_JOINTS,
      input_fps=input_fps,
      input_file=input_file,
      output_fps=output_fps,
      output_name=output_name,
      render=render,
      line_range=line_range,
      renderer=renderer,
    )
  finally:
    if temp_path is not None:
      os.unlink(temp_path)


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
