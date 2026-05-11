#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import zmq

from mujoco_pi_plus_sim.gait_policy import PolicyGaitController


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="External gait node: direct sim loop + gait + override fusion")
    p.add_argument("--sim-host", default="127.0.0.1")
    p.add_argument("--sim-port", type=int, default=5555)
    p.add_argument("--sensor-bind", default="tcp://127.0.0.1:6001", help="PUB bind for sim_tick stream")
    p.add_argument("--intent-bind", default="tcp://127.0.0.1:6002", help="SUB bind for joint_override input")
    p.add_argument("--policy", type=Path, default=Path("assets/policies/pi_plus_model_40000.pt"))
    p.add_argument("--policy-device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--vx", type=float, default=0.0)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--wz", type=float, default=0.0)
    p.add_argument("--cmd-connect", default="tcp://127.0.0.1:6003", help="SUB endpoint for runtime cmd_vel updates")
    p.add_argument("--rids", default="", help="Comma-separated robot IDs to control, e.g. 0,1,7. Empty means all.")
    p.add_argument(
        "--rewrite-cmd-in-obs",
        action="store_true",
        help="Rewrite cmd slots in obs history from external cmd stream (debug only; disables strict internal parity).",
    )
    p.add_argument("--team", choices=["red", "blue", "both"], default="both", help="Team filter for controlled robots.")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--source", default="gait_node")
    return p.parse_args()


def _recv_all(sub: zmq.Socket) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    while True:
        try:
            msgs.append(sub.recv_json(flags=zmq.NOBLOCK))
        except zmq.Again:
            break
    return msgs


def main() -> None:
    args = parse_args()
    controller = PolicyGaitController(args.policy, args.policy_device)
    cmd = np.asarray([args.vx, args.vy, args.wz], dtype=np.float32)
    obs_step_dim = 69
    rid_filter = {
        int(x.strip()) for x in str(args.rids).split(",") if x.strip() != ""
    }

    ctx = zmq.Context()
    sim = ctx.socket(zmq.REQ)
    sim.connect(f"tcp://{args.sim_host}:{args.sim_port}")

    cmd_sub = ctx.socket(zmq.SUB)
    cmd_sub.connect(args.cmd_connect)
    cmd_sub.setsockopt_string(zmq.SUBSCRIBE, "")

    intent_sub = ctx.socket(zmq.SUB)
    intent_sub.bind(args.intent_bind)
    intent_sub.setsockopt_string(zmq.SUBSCRIBE, "")

    sensor_pub = ctx.socket(zmq.PUB)
    sensor_pub.bind(args.sensor_bind)

    joint_overrides: dict[int, dict[int, float]] = {}
    current_seq = -1
    # Prime one step to get initial sensors/state.
    sim.send_json({"timestamp": time.time(), "source": args.source, "joint_actions": [], "commands": []})
    resp = sim.recv_json()

    while True:
        step_t0 = time.time()
        while True:
            try:
                m = cmd_sub.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            if m.get("type") == "cmd_vel":
                try:
                    cmd[:] = [float(m.get("vx", cmd[0])), float(m.get("vy", cmd[1])), float(m.get("wz", cmd[2]))]
                except Exception:
                    pass

        for m in _recv_all(intent_sub):
            if m.get("type") != "joint_override":
                continue
            rid = int(m.get("rid", -1))
            ov = m.get("overrides")
            if rid < 0 or not isinstance(ov, dict):
                continue
            parsed: dict[int, float] = {}
            for k, v in ov.items():
                try:
                    parsed[int(k)] = float(v)
                except Exception:
                    continue
            joint_overrides[rid] = parsed

        robots = resp.get("sensors", {}).get("robots", [])
        outgoing: list[dict[str, Any]] = []
        for r in robots:
            rid = int(r.get("id", -1))
            team = str(r.get("team", ""))
            n = int(len(r.get("joint_pos_target", [])))
            obs = np.asarray(r.get("obs", []), dtype=np.float32)
            if rid < 0 or n <= 0 or obs.size == 0:
                continue
            if rid_filter and rid not in rid_filter:
                continue
            if args.team != "both" and team != args.team:
                continue
            if args.rewrite_cmd_in_obs and obs.size % obs_step_dim == 0:
                n_steps = obs.size // obs_step_dim
                for i in range(n_steps):
                    base = i * obs_step_dim
                    obs[base + 6 : base + 9] = cmd
            a = controller.infer_actions(obs.reshape(1, -1))[0]
            if a.shape[0] != n:
                a = np.resize(a, (n,)).astype(np.float32)
            else:
                a = a.astype(np.float32, copy=False)
            for idx, val in joint_overrides.get(rid, {}).items():
                if 0 <= idx < n:
                    a[idx] = float(val)
            outgoing.append({"id": rid, "a": a.tolist()})

        commands = [{"id": int(r.get("id", -1)), "cmd": [float(cmd[0]), float(cmd[1]), float(cmd[2])]} for r in robots if int(r.get("id", -1)) >= 0]
        req = {
            "timestamp": time.time(),
            "source": args.source,
            "joint_actions": outgoing,
            "commands": commands,
        }
        sim.send_json(req)
        resp = sim.recv_json()

        sensor_pub.send_json(
            {
                "type": "sim_tick",
                "timestamp": time.time(),
                "seq": current_seq + 1,
                "response": resp,
            }
        )
        current_seq += 1

        dt_left = float(args.dt) - (time.time() - step_t0)
        if dt_left > 0:
            time.sleep(dt_left)


if __name__ == "__main__":
    main()
