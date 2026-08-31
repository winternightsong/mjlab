#!/usr/bin/env python3
"""Smoothly slow a local time segment in an mjlab motion NPZ."""

from __future__ import annotations

import argparse

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate((q[..., 1:], q[..., :1]), axis=-1)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate((q[..., -1:], q[..., :-1]), axis=-1)


def smoothstep5(x: np.ndarray) -> np.ndarray:
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def angular_velocity_wxyz(q: np.ndarray, fps: float) -> np.ndarray:
    xyzw = wxyz_to_xyzw(q.reshape(-1, 4)).reshape(q.shape)
    matrices = Rotation.from_quat(xyzw.reshape(-1, 4)).as_matrix().reshape(q.shape[:-1] + (3, 3))
    relative = np.einsum("...ji,...jk->...ik", matrices[:-1], matrices[1:])
    local = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(relative.shape[:-2] + (3,))
    world = np.einsum("...ij,...j->...i", matrices[:-1], local) * fps
    result = np.empty(q.shape[:-1] + (3,), dtype=np.float32)
    result[:-1] = world.astype(np.float32)
    result[-1] = result[-2]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--start", type=float, default=117.0)
    parser.add_argument("--end", type=float, default=124.0)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--ramp", type=float, default=1.0)
    args = parser.parse_args()

    if not 0.0 < args.speed <= 1.0:
        raise ValueError("speed must be in (0, 1]")
    src = np.load(args.source)
    data = {key: src[key].copy() for key in src.files}
    fps = float(data["fps"][0])
    frame_count = len(data["joint_pos"])
    old_t = np.arange(frame_count, dtype=np.float64) / fps

    # d(new_time)/d(old_time): 2.0 means the source motion runs at 0.5 speed.
    full_scale = 1.0 / args.speed
    scale = np.ones_like(old_t)
    ramp_in = (old_t >= args.start - args.ramp) & (old_t < args.start)
    ramp_out = (old_t > args.end) & (old_t <= args.end + args.ramp)
    scale[(old_t >= args.start) & (old_t <= args.end)] = full_scale
    x_in = (old_t[ramp_in] - (args.start - args.ramp)) / args.ramp
    x_out = (old_t[ramp_out] - args.end) / args.ramp
    scale[ramp_in] = 1.0 + (full_scale - 1.0) * smoothstep5(x_in)
    scale[ramp_out] = full_scale - (full_scale - 1.0) * smoothstep5(x_out)

    warped_t = np.zeros_like(old_t)
    warped_t[1:] = np.cumsum(0.5 * (scale[:-1] + scale[1:]) / fps)
    new_t = np.arange(round(warped_t[-1] * fps) + 1, dtype=np.float64) / fps
    source_t = np.interp(new_t, warped_t, old_t)

    for key in ("joint_pos", "body_pos_w"):
        shape = data[key].shape
        flat = data[key].reshape(frame_count, -1)
        data[key] = PchipInterpolator(old_t, flat, axis=0)(source_t).reshape((len(new_t),) + shape[1:]).astype(np.float32)

    shape = data["body_quat_w"].shape
    quats = wxyz_to_xyzw(data["body_quat_w"].reshape(frame_count, -1, 4))
    output = np.empty((len(new_t), quats.shape[1], 4), dtype=np.float32)
    for body in range(quats.shape[1]):
        output[:, body] = xyzw_to_wxyz(Slerp(old_t, Rotation.from_quat(quats[:, body]))(source_t).as_quat())
    data["body_quat_w"] = output.reshape((len(new_t),) + shape[1:])

    data["joint_vel"] = np.gradient(data["joint_pos"], 1.0 / fps, axis=0).astype(np.float32)
    data["body_lin_vel_w"] = np.gradient(data["body_pos_w"], 1.0 / fps, axis=0).astype(np.float32)
    data["body_ang_vel_w"] = angular_velocity_wxyz(data["body_quat_w"], fps)
    data["root_lin_vel_w"] = data["body_lin_vel_w"][:, 0].copy()
    data["root_ang_vel_w"] = data["body_ang_vel_w"][:, 0].copy()

    dynamic = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "root_lin_vel_w", "root_ang_vel_w")
    for key in dynamic:
        if len(data[key]) != len(new_t) or not np.isfinite(data[key]).all():
            raise RuntimeError(f"Invalid output field {key}: {data[key].shape}")
    qnorm = np.linalg.norm(data["body_quat_w"], axis=-1)
    if np.max(np.abs(qnorm - 1.0)) > 1e-4:
        raise RuntimeError("Quaternion normalization check failed")

    np.savez_compressed(args.destination, **data)
    mapped_start = float(np.interp(args.start, old_t, warped_t))
    mapped_end = float(np.interp(args.end, old_t, warped_t))
    print(f"frames {frame_count} -> {len(new_t)}")
    print(f"duration {old_t[-1]:.3f}s -> {new_t[-1]:.3f}s")
    print(f"source core {args.start:.3f}-{args.end:.3f}s -> output {mapped_start:.3f}-{mapped_end:.3f}s")
    print(f"effective core speed={args.speed:.3f}, ramps={args.ramp:.3f}s each")
    core = (new_t >= mapped_start) & (new_t <= mapped_end)
    print("core max joint velocity", float(np.max(np.abs(data["joint_vel"][core]))))
    print("core max root angular velocity", float(np.max(np.linalg.norm(data["root_ang_vel_w"][core], axis=-1))))


if __name__ == "__main__":
    main()
