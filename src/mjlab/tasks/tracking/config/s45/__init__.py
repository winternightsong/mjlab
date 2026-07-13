from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import kuavo_s45_flat_tracking_env_cfg
from .rl_cfg import kuavo_s45_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-KUAVO-S45",
  env_cfg=kuavo_s45_flat_tracking_env_cfg(),
  play_env_cfg=kuavo_s45_flat_tracking_env_cfg(play=True),
  rl_cfg=kuavo_s45_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-KUAVO-S45-No-State-Estimation",
  env_cfg=kuavo_s45_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=kuavo_s45_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=kuavo_s45_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
