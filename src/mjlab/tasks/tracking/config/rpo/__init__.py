"""RPO BeyondMimic-style tracking tasks."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import rpo_flat_tracking_env_cfg
from .rl_cfg import rpo_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-RPO",
  env_cfg=rpo_flat_tracking_env_cfg(),
  play_env_cfg=rpo_flat_tracking_env_cfg(play=True),
  rl_cfg=rpo_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
