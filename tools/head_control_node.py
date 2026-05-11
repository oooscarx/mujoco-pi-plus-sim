#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time

import zmq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Head control node: emit joint overrides")
    p.add_argument("--sensor-connect", default="tcp://127.0.0.1:6001")
    p.add_argument("--intent-connect", default="tcp://127.0.0.1:6002")
    p.add_argument("--rid", type=int, default=0)
    p.add_argument("--yaw-index", type=int, default=0, help="joint index in action vector")
    p.add_argument("--pitch-index", type=int, default=1, help="joint index in action vector")
    p.add_argument("--yaw-amp", type=float, default=0.35)
    p.add_argument("--pitch", type=float, default=0.0)
    p.add_argument("--scan-hz", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ctx = zmq.Context()

    sub = ctx.socket(zmq.SUB)
    sub.connect(args.sensor_connect)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")

    pub = ctx.socket(zmq.PUB)
    pub.connect(args.intent_connect)

    t0 = time.time()
    while True:
        _ = sub.recv_json()
        t = time.time() - t0
        yaw = args.yaw_amp * math.sin(2.0 * math.pi * args.scan_hz * t)
        pub.send_json(
            {
                "type": "joint_override",
                "rid": args.rid,
                "overrides": {
                    str(args.yaw_index): float(yaw),
                    str(args.pitch_index): float(args.pitch),
                },
            }
        )


if __name__ == "__main__":
    main()
