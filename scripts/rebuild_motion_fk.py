#!/usr/bin/env python3
"""Rebuild motion body states from root pose and joints using MuJoCo FK."""

from __future__ import annotations

import argparse

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate((q[..., 1:], q[..., :1]), axis=-1)


def angular_velocity_wxyz(q: np.ndarray, fps: float) -> np.ndarray:
    xyzw = wxyz_to_xyzw(q.reshape(-1, 4)).reshape(q.shape)
    rotations = Rotation.from_quat(xyzw.reshape(-1, 4))
    mats = rotations.as_matrix().reshape(q.shape[:-1] + (3, 3))
    rel = np.einsum("...ji,...jk->...ik", mats[:-1], mats[1:])
    rv_local = Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(
        rel.shape[:-2] + (3,)
    )
    rv_world = np.einsum("...ij,...j->...i", mats[:-1], rv_local) * fps
    out = np.empty(q.shape[:-1] + (3,), dtype=np.float32)
    out[:-1] = rv_world.astype(np.float32)
    out[-1] = out[-2]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()

    src = np.load(args.source)
    data = {key: src[key].copy() for key in src.files}
    fps = float(data["fps"][0])
    model = mujoco.MjModel.from_xml_path(args.model)
    mjdata = mujoco.MjData(model)

    joint_names = [
        *[f"leg_l{i}_joint" for i in range(1, 7)],
        *[f"leg_r{i}_joint" for i in range(1, 7)],
        *[f"zarm_l{i}_joint" for i in range(1, 8)],
        *[f"zarm_r{i}_joint" for i in range(1, 8)],
    ]
    body_names = [str(name) for name in data["body_names"]]
    joint_qpos_adrs = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in joint_names
    ]
    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names
    ]
    if any(index < 0 for index in body_ids):
        raise RuntimeError("Motion body names do not all exist in the MuJoCo model")
    if data["joint_pos"].shape[1] != len(joint_qpos_adrs):
        raise RuntimeError("Motion joint count does not match the 26-DoF S45 mapping")

    root_pos = data["body_pos_w"][:, 0].copy()
    root_quat = data["body_quat_w"][:, 0].copy()
    body_pos = np.empty_like(data["body_pos_w"])
    body_quat = np.empty_like(data["body_quat_w"])
    for frame in range(len(data["joint_pos"])):
        mujoco.mj_resetData(model, mjdata)
        mjdata.qpos[:3] = root_pos[frame]
        mjdata.qpos[3:7] = root_quat[frame]
        mjdata.qpos[joint_qpos_adrs] = data["joint_pos"][frame]
        mujoco.mj_forward(model, mjdata)
        body_pos[frame] = mjdata.xpos[body_ids]
        body_quat[frame] = mjdata.xquat[body_ids]

    data["body_pos_w"] = body_pos
    data["body_quat_w"] = body_quat
    data["joint_vel"] = np.gradient(data["joint_pos"], 1.0 / fps, axis=0).astype(np.float32)
    data["body_lin_vel_w"] = np.gradient(body_pos, 1.0 / fps, axis=0).astype(np.float32)
    data["body_ang_vel_w"] = angular_velocity_wxyz(body_quat, fps)
    data["root_lin_vel_w"] = data["body_lin_vel_w"][:, 0].copy()
    data["root_ang_vel_w"] = data["body_ang_vel_w"][:, 0].copy()

    for key, value in data.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise RuntimeError(f"Non-finite values in {key}")
    np.savez_compressed(args.destination, **data)
    print(f"frames={len(body_pos)} duration={len(body_pos) / fps:.3f}s")


if __name__ == "__main__":
    main()
