#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

import zmq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic timing driver for internal policy mode")
    p.add_argument("--sim-host", default="127.0.0.1")
    p.add_argument("--sim-port", type=int, default=5555)
    p.add_argument("--dt", type=float, default=0.002, help="Target step period in seconds (sim2sim uses 0.002)")
    p.add_argument("--duration", type=float, default=20.0, help="Run duration in seconds")
    p.add_argument("--rids", default="0,7", help="Comma-separated robot IDs")
    p.add_argument("--vx", type=float, default=1.0)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--wz", type=float, default=0.0)
    p.add_argument("--source", default="policy_timing_driver")
    p.add_argument("--print-every", type=int, default=500, help="Print status every N ticks")
    return p.parse_args()


def _parse_rids(s: str) -> list[int]:
    out: list[int] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def _summarize(xs: list[float]) -> tuple[float, float, float]:
    if not xs:
        return 0.0, 0.0, 0.0
    return float(min(xs)), float(statistics.mean(xs)), float(max(xs))


def main() -> None:
    args = parse_args()
    rids = _parse_rids(args.rids)
    if not rids:
        raise ValueError("No valid robot ids")

    ctx = zmq.Context()
    req = ctx.socket(zmq.REQ)
    req.connect(f"tcp://{args.sim_host}:{args.sim_port}")

    t0 = time.monotonic()
    next_tick = t0
    end_t = t0 + max(0.1, float(args.duration))

    intervals_ms: list[float] = []
    latencies_ms: list[float] = []
    overruns_ms: list[float] = []
    last_tick_sent: float | None = None
    tick = 0

    while True:
        now = time.monotonic()
        if now >= end_t:
            break

        sleep_s = next_tick - now
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            overruns_ms.append((-sleep_s) * 1000.0)

        send_t = time.monotonic()
        if last_tick_sent is not None:
            intervals_ms.append((send_t - last_tick_sent) * 1000.0)
        last_tick_sent = send_t

        wall_ts = time.time()
        msg: dict[str, Any] = {
            "timestamp": wall_ts,
            "source": args.source,
            "commands": [
                {
                    "id": rid,
                    "cmd": [float(args.vx), float(args.vy), float(args.wz)],
                    "timestamp": wall_ts,
                    "source": args.source,
                }
                for rid in rids
            ],
        }
        req.send_json(msg)
        _ = req.recv_json()
        recv_t = time.monotonic()
        latencies_ms.append((recv_t - send_t) * 1000.0)

        tick += 1
        if args.print_every > 0 and tick % args.print_every == 0:
            i_min, i_mean, i_max = _summarize(intervals_ms[-args.print_every :])
            l_min, l_mean, l_max = _summarize(latencies_ms[-args.print_every :])
            print(
                f"[policy_timing_driver] tick={tick} interval_ms(min/mean/max)={i_min:.3f}/{i_mean:.3f}/{i_max:.3f} "
                f"rtt_ms(min/mean/max)={l_min:.3f}/{l_mean:.3f}/{l_max:.3f}",
                flush=True,
            )

        next_tick += float(args.dt)

    i_min, i_mean, i_max = _summarize(intervals_ms)
    l_min, l_mean, l_max = _summarize(latencies_ms)
    o_min, o_mean, o_max = _summarize(overruns_ms)
    print(
        "[policy_timing_driver] done "
        f"ticks={tick} interval_ms(min/mean/max)={i_min:.3f}/{i_mean:.3f}/{i_max:.3f} "
        f"rtt_ms(min/mean/max)={l_min:.3f}/{l_mean:.3f}/{l_max:.3f} "
        f"overrun_ms(min/mean/max)={o_min:.3f}/{o_mean:.3f}/{o_max:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
