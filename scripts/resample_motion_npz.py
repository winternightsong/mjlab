"""Resample an mjlab motion NPZ and optionally change playback speed."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def interpolate_linear(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
  left = np.floor(indices).astype(np.int64)
  right = np.minimum(left + 1, values.shape[0] - 1)
  alpha = (indices - left).reshape((-1,) + (1,) * (values.ndim - 1))
  return values[left] * (1.0 - alpha) + values[right] * alpha


def interpolate_quaternions(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
  """Shortest-path normalized interpolation for wxyz quaternions."""
  left = np.floor(indices).astype(np.int64)
  right = np.minimum(left + 1, values.shape[0] - 1)
  q0 = values[left]
  q1 = values[right]
  q1 = np.where(np.sum(q0 * q1, axis=-1, keepdims=True) < 0.0, -q1, q1)
  alpha = (indices - left).reshape((-1,) + (1,) * (values.ndim - 1))
  result = q0 * (1.0 - alpha) + q1 * alpha
  return result / np.linalg.norm(result, axis=-1, keepdims=True).clip(1.0e-8)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("input", type=Path)
  parser.add_argument("output", type=Path)
  parser.add_argument("--output-fps", type=float, default=50.0)
  parser.add_argument(
    "--speed",
    type=float,
    default=0.8,
    help="Playback speed relative to the source (0.8 is 20%% slower).",
  )
  args = parser.parse_args()
  if args.output_fps <= 0.0 or args.speed <= 0.0:
    raise ValueError("output-fps and speed must be positive")

  source = np.load(args.input, allow_pickle=True)
  source_fps = float(np.asarray(source["fps"]).reshape(-1)[0])
  frame_count = source["joint_pos"].shape[0]
  source_duration = (frame_count - 1) / source_fps
  output_duration = source_duration / args.speed
  output_frames = int(round(output_duration * args.output_fps)) + 1
  source_indices = (
    np.arange(output_frames, dtype=np.float64)
    / args.output_fps
    * args.speed
    * source_fps
  ).clip(0.0, frame_count - 1)

  result: dict[str, np.ndarray] = {}
  quaternion_keys = {"body_quat_w"}
  velocity_keys = {
    "joint_vel",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "root_lin_vel_w",
    "root_ang_vel_w",
  }
  for key in source.files:
    values = source[key]
    if key == "fps":
      result[key] = np.asarray([args.output_fps], dtype=values.dtype)
    elif values.ndim > 0 and values.shape[0] == frame_count:
      if key in quaternion_keys:
        result[key] = interpolate_quaternions(values, source_indices).astype(
          values.dtype
        )
      else:
        result[key] = interpolate_linear(values, source_indices).astype(values.dtype)
      if key in velocity_keys:
        result[key] *= args.speed
    else:
      result[key] = values

  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(args.output, **result)
  print(f"source: {frame_count} frames @ {source_fps:g} Hz ({source_duration:.3f}s)")
  print(
    f"output: {output_frames} frames @ {args.output_fps:g} Hz "
    f"({output_duration:.3f}s), speed={args.speed:g}x"
  )


if __name__ == "__main__":
  main()
