#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import zmq

from mujoco_pi_plus_sim.gait_policy import PolicyGaitController


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="External gait controller over ZMQ")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--policy", type=Path, default=Path("assets/policies/pi_plus_model_40000.pt"))
    p.add_argument("--policy-device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--dt", type=float, default=0.02, help="controller loop period (sec)")
    p.add_argument("--source", default="external_gait")
    p.add_argument("--vx", type=float, default=0.0, help="desired forward cmd for gait policy")
    p.add_argument("--vy", type=float, default=0.0, help="desired lateral cmd for gait policy")
    p.add_argument("--wz", type=float, default=0.0, help="desired yaw-rate cmd for gait policy")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    controller = PolicyGaitController(args.policy, args.policy_device)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://{args.host}:{args.port}")

    # Prime once to get sensors.
    sock.send_json({"timestamp": time.time(), "source": args.source})
    resp = sock.recv_json()
    cmd = np.asarray([args.vx, args.vy, args.wz], dtype=np.float32)
    obs_step_dim = 69  # pi_plus obs step size in this repo

    while True:
        sensors = resp.get("sensors", {})
        robots = sensors.get("robots", [])

        joint_actions = []
        for r in robots:
            rid = int(r.get("id", -1))
            obs = np.asarray(r.get("obs", []), dtype=np.float32)
            if rid < 0 or obs.size == 0:
                continue
            # Externalize cmd control: inject cmd terms into every history step.
            if obs.size % obs_step_dim == 0:
                for base in range(0, obs.size, obs_step_dim):
                    obs[base + 6 : base + 9] = cmd
            act = controller.infer_actions(obs.reshape(1, -1))[0]
            joint_actions.append({"id": rid, "a": act.tolist()})

        req = {
            "timestamp": time.time(),
            "source": args.source,
            "joint_actions": joint_actions,
        }
        sock.send_json(req)
        resp = sock.recv_json()
        if args.dt > 0:
            time.sleep(args.dt)


if __name__ == "__main__":
    main()
