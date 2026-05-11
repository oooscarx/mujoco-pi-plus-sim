#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import os
from pathlib import Path

from .runtime_config import parse_runtime_args

# src/mujoco_pi_plus_sim/runner.py -> project root is two levels up from package dir.
MUJOCO_DIR = Path(__file__).resolve().parents[2]


def main():
    args = parse_runtime_args(MUJOCO_DIR)
    # Keep Linux default for headless servers; allow explicit per-run override.
    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")
    from .multi_robot_sim import run_sim

    template_dir = MUJOCO_DIR / "web" / "templates"
    run_sim(args=args, template_dir=template_dir)


if __name__ == "__main__":
    main()
