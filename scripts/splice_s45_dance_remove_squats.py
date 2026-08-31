#!/usr/bin/env python3
"""Remove two squat sections from an mjlab motion NPZ with smooth joins."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class Cut:
    left_s: float
    right_s: float
    transition_s: float


CUTS = (Cut(171.92, 184.60, 1.20), Cut(190.20, 197.10, 1.20))


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate((q[..., 1:], q[..., :1]), axis=-1)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate((q[..., -1:], q[..., :-1]), axis=-1)


def quat_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def yaw_from_wxyz(q: np.ndarray) -> float:
    return float(Rotation.from_quat(wxyz_to_xyzw(q)).as_euler("xyz")[2])


def align_suffix(data: dict[str, np.ndarray], left: int, right: int) -> None:
    """Rigidly align the suffix root XY/yaw and Z to the left splice state."""
    root_pos = data["body_pos_w"][:, 0]
    root_quat = data["body_quat_w"][:, 0]
    dyaw = yaw_from_wxyz(root_quat[left]) - yaw_from_wxyz(root_quat[right])
    c, s = np.cos(dyaw), np.sin(dyaw)
    rot = np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)
    pivot = root_pos[right].astype(np.float64)
    target = root_pos[left].astype(np.float64)

    pos = data["body_pos_w"][right:].astype(np.float64)
    data["body_pos_w"][right:] = ((pos - pivot) @ rot.T + target).astype(np.float32)

    q_delta = np.array((np.cos(dyaw / 2), 0.0, 0.0, np.sin(dyaw / 2)), dtype=np.float64)
    q_delta = np.broadcast_to(q_delta, data["body_quat_w"][right:].shape)
    data["body_quat_w"][right:] = quat_multiply_wxyz(
        q_delta, data["body_quat_w"][right:].astype(np.float64)
    ).astype(np.float32)


def smoothstep5(x: np.ndarray) -> np.ndarray:
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def interpolate_quat(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    flat0 = q0.reshape(-1, 4)
    flat1 = q1.reshape(-1, 4)
    out = np.empty((len(alpha), len(flat0), 4), dtype=np.float64)
    for k in range(len(flat0)):
        key = Rotation.from_quat(wxyz_to_xyzw(np.stack((flat0[k], flat1[k]))))
        out[:, k] = xyzw_to_wxyz(Slerp((0.0, 1.0), key)(alpha).as_quat())
    return out.reshape((len(alpha),) + q0.shape).astype(np.float32)


def transition(data: dict[str, np.ndarray], left: int, right: int, count: int) -> dict[str, np.ndarray]:
    # Interior samples only: the prefix supplies alpha=0 and suffix alpha=1.
    x = np.arange(1, count, dtype=np.float64) / count
    a = smoothstep5(x)
    result: dict[str, np.ndarray] = {}
    for key in ("joint_pos", "body_pos_w"):
        v0, v1 = data[key][left], data[key][right]
        shape = (len(a),) + (1,) * v0.ndim
        aa = a.reshape(shape)
        result[key] = ((1.0 - aa) * v0 + aa * v1).astype(np.float32)
    result["body_quat_w"] = interpolate_quat(
        data["body_quat_w"][left], data["body_quat_w"][right], a
    )
    return result


def angular_velocity_wxyz(q: np.ndarray, fps: float) -> np.ndarray:
    xyzw = wxyz_to_xyzw(q.reshape(-1, 4)).reshape(q.shape)
    rotations = Rotation.from_quat(xyzw.reshape(-1, 4))
    # Relative rotation from sample t to t+1, expressed as a rotation vector.
    mats = rotations.as_matrix().reshape(q.shape[:-1] + (3, 3))
    rel = np.einsum("...ji,...jk->...ik", mats[:-1], mats[1:])
    rv_local = Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(rel.shape[:-2] + (3,))
    # Convert local angular velocity to world coordinates, matching *_ang_vel_w.
    rv_world = np.einsum("...ij,...j->...i", mats[:-1], rv_local) * fps
    out = np.empty(q.shape[:-1] + (3,), dtype=np.float32)
    out[:-1] = rv_world.astype(np.float32)
    out[-1] = out[-2]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()

    src = np.load(args.source)
    data = {key: src[key].copy() for key in src.files}
    fps = float(data["fps"][0])
    cuts = [(round(c.left_s * fps), round(c.right_s * fps), round(c.transition_s * fps)) for c in CUTS]

    # Align in chronological order. The second alignment sees the coordinate
    # transform already applied by the first one, keeping the final suffix continuous.
    for left, right, _ in cuts:
        align_suffix(data, left, right)

    dynamic = {"joint_pos", "body_pos_w", "body_quat_w"}
    assembled: dict[str, list[np.ndarray]] = {key: [] for key in dynamic}
    cursor = 0
    new_join_frames = []
    output_count = 0
    for left, right, count in cuts:
        for key in dynamic:
            assembled[key].append(data[key][cursor : left + 1])
        output_count += left + 1 - cursor
        tr = transition(data, left, right, count)
        for key in dynamic:
            assembled[key].append(tr[key])
        output_count += count - 1
        new_join_frames.append(output_count - 1)
        cursor = right
    for key in dynamic:
        assembled[key].append(data[key][cursor:])
        data[key] = np.concatenate(assembled[key], axis=0)

    # Recompute every velocity from the final, transformed trajectory.
    data["joint_vel"] = np.gradient(data["joint_pos"], 1.0 / fps, axis=0).astype(np.float32)
    data["body_lin_vel_w"] = np.gradient(data["body_pos_w"], 1.0 / fps, axis=0).astype(np.float32)
    data["body_ang_vel_w"] = angular_velocity_wxyz(data["body_quat_w"], fps)
    data["root_lin_vel_w"] = data["body_lin_vel_w"][:, 0].copy()
    data["root_ang_vel_w"] = data["body_ang_vel_w"][:, 0].copy()

    lengths = {key: len(data[key]) for key in dynamic}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"Mismatched output lengths: {lengths}")
    for key, value in data.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise RuntimeError(f"Non-finite values in {key}")
    qnorm = np.linalg.norm(data["body_quat_w"], axis=-1)
    if np.max(np.abs(qnorm - 1.0)) > 1e-4:
        raise RuntimeError("Quaternion normalization check failed")

    np.savez_compressed(args.destination, **data)
    print(f"source_frames={len(src['joint_pos'])} output_frames={len(data['joint_pos'])}")
    print(f"source_duration={len(src['joint_pos']) / fps:.3f}s output_duration={len(data['joint_pos']) / fps:.3f}s")
    print(f"join_frames={new_join_frames} join_times={[round(x / fps, 3) for x in new_join_frames]}")
    for frame in new_join_frames:
        sl = slice(max(0, frame - 80), min(len(data["joint_pos"]), frame + 81))
        print(
            "join",
            frame,
            "max_joint_vel",
            float(np.max(np.abs(data["joint_vel"][sl]))),
            "max_root_lin_vel",
            float(np.max(np.linalg.norm(data["root_lin_vel_w"][sl], axis=-1))),
            "max_root_ang_vel",
            float(np.max(np.linalg.norm(data["root_ang_vel_w"][sl], axis=-1))),
        )


if __name__ == "__main__":
    main()
