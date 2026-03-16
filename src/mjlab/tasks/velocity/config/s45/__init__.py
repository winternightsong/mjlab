from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  kuavo_s45_flat_env_cfg,
  kuavo_s45_rough_env_cfg,
)
from .rl_cfg import kuavo_s45_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-KUAVO-S45",
  env_cfg=kuavo_s45_rough_env_cfg(),
  play_env_cfg=kuavo_s45_rough_env_cfg(play=True),
  rl_cfg=kuavo_s45_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-KUAVO-S45",
  env_cfg=kuavo_s45_flat_env_cfg(),
  play_env_cfg=kuavo_s45_flat_env_cfg(play=True),
  rl_cfg=kuavo_s45_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
