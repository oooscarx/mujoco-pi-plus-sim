#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import zmq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publish runtime cmd_vel updates for gait_node")
    p.add_argument("--bind", default="tcp://127.0.0.1:6003")
    p.add_argument("--vx", type=float, default=0.0)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--wz", type=float, default=0.0)
    p.add_argument("--rate", type=float, default=10.0, help="publish rate in Hz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(args.bind)
    period = 1.0 / max(0.1, float(args.rate))

    while True:
        pub.send_json(
            {
                "type": "cmd_vel",
                "vx": float(args.vx),
                "vy": float(args.vy),
                "wz": float(args.wz),
                "timestamp": time.time(),
            }
        )
        time.sleep(period)


if __name__ == "__main__":
    main()
