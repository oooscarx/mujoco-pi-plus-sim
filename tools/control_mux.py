#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import zmq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Control mux: merge intents and drive MuJoCo ZMQ")
    p.add_argument("--sim-host", default="127.0.0.1")
    p.add_argument("--sim-port", type=int, default=5555)
    p.add_argument("--intent-bind", default="tcp://127.0.0.1:6002", help="SUB bind for control intents")
    p.add_argument("--sensor-bind", default="tcp://127.0.0.1:6001", help="PUB bind for sensor/state stream")
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--source", default="control_mux")
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
    ctx = zmq.Context()

    sim = ctx.socket(zmq.REQ)
    sim.connect(f"tcp://{args.sim_host}:{args.sim_port}")

    intent_sub = ctx.socket(zmq.SUB)
    intent_sub.bind(args.intent_bind)
    intent_sub.setsockopt_string(zmq.SUBSCRIBE, "")

    sensor_pub = ctx.socket(zmq.PUB)
    sensor_pub.bind(args.sensor_bind)

    gait_actions_by_seq: dict[int, dict[int, np.ndarray]] = {}
    gait_latest_by_rid: dict[int, np.ndarray] = {}
    joint_overrides: dict[int, dict[int, float]] = {}
    current_seq = -1

    # Prime one step to get initial sensors/state.
    sim.send_json({"timestamp": time.time(), "source": args.source, "joint_actions": []})
    resp = sim.recv_json()

    while True:
        step_t0 = time.time()

        for m in _recv_all(intent_sub):
            t = m.get("type")
            if t == "gait_action":
                rid = int(m.get("rid", -1))
                a = m.get("a")
                seq = int(m.get("seq", -1))
                if rid >= 0 and isinstance(a, list) and seq >= 0:
                    arr = np.asarray(a, dtype=np.float32)
                    gait_actions_by_seq.setdefault(seq, {})[rid] = arr
                    gait_latest_by_rid[rid] = arr
            elif t == "joint_override":
                rid = int(m.get("rid", -1))
                ov = m.get("overrides")
                if rid >= 0 and isinstance(ov, dict):
                    parsed: dict[int, float] = {}
                    for k, v in ov.items():
                        try:
                            parsed[int(k)] = float(v)
                        except Exception:
                            continue
                    joint_overrides[rid] = parsed

        robots = resp.get("sensors", {}).get("robots", [])
        outgoing = []
        for r in robots:
            rid = int(r.get("id", -1))
            if rid < 0:
                continue
            n = len(r.get("joint_pos_target", []))
            if n <= 0:
                continue
            seq_actions = gait_actions_by_seq.get(current_seq, {})
            if rid in seq_actions:
                a = seq_actions[rid].copy()
            elif rid in gait_latest_by_rid:
                a = gait_latest_by_rid[rid].copy()
            else:
                a = np.zeros((n,), dtype=np.float32)
            if a.shape[0] != n:
                a = np.resize(a, (n,)).astype(np.float32)
            for idx, val in joint_overrides.get(rid, {}).items():
                if 0 <= idx < n:
                    a[idx] = float(val)
            outgoing.append({"id": rid, "a": a.tolist()})

        req = {
            "timestamp": time.time(),
            "source": args.source,
            "joint_actions": outgoing,
        }
        sim.send_json(req)
        resp = sim.recv_json()

        sensor_pub.send_json({
            "type": "sim_tick",
            "timestamp": time.time(),
            "seq": current_seq + 1,
            "response": resp,
        })
        current_seq += 1
        # Keep only recent action windows.
        stale = [k for k in gait_actions_by_seq.keys() if k < current_seq - 2]
        for k in stale:
            gait_actions_by_seq.pop(k, None)

        dt_left = args.dt - (time.time() - step_t0)
        if dt_left > 0:
            time.sleep(dt_left)


if __name__ == "__main__":
    main()
