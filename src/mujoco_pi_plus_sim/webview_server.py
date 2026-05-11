# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import base64
import threading
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO
from PIL import Image


@dataclass
class WebMsgBuffer:
    reset_env: bool = False
    restart_match: bool = False
    viewer_point: list[float] | None = None
    viewer_look_at: list[float] | None = None
    camera_preset: str | None = None
    teleport_cmd: tuple[str, float, float, float | None, float | None] | None = None
    spawn_points: dict[str, list[float]] | None = None
    velocity_cmds: list[tuple[str, float, float, float]] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class MujocoLabWebView:
    def __init__(self, template_dir: Path, allow_keyboard_control: bool = False):
        self.app = Flask(__name__, template_folder=str(template_dir))
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.msg = WebMsgBuffer()
        self.allow_keyboard_control = bool(allow_keyboard_control)
        self._field_meta: dict | None = None
        self._setup_routes_and_events()

    def _setup_routes_and_events(self):
        @self.app.route("/")
        def index():
            return render_template("index.html", allow_keyboard_control=self.allow_keyboard_control)

        @self.socketio.on("connect")
        def on_connect():
            print("[MujocoWebView] Client connected")
            if self._field_meta is not None:
                self.socketio.emit("field_meta", self._field_meta)

        @self.socketio.on("reset_env")
        def on_reset():
            with self.msg.lock:
                self.msg.reset_env = True

        @self.socketio.on("restart_match")
        def on_restart_match():
            with self.msg.lock:
                self.msg.restart_match = True

        @self.socketio.on("set_viewer_point")
        def on_view_point(data):
            with self.msg.lock:
                self.msg.viewer_point = data.get("point", [3.0, 3.0, 1.0])

        @self.socketio.on("set_viewer_look_at")
        def on_view_look(data):
            with self.msg.lock:
                self.msg.viewer_look_at = data.get("point", [0.0, 0.0, 1.0])

        @self.socketio.on("set_camera_preset")
        def on_camera_preset(data):
            with self.msg.lock:
                self.msg.camera_preset = data.get("preset", "Top")

        @self.socketio.on("teleport_entity")
        def on_teleport(data):
            with self.msg.lock:
                self.msg.teleport_cmd = (
                    data.get("name", ""),
                    float(data.get("x", 0.0)),
                    float(data.get("y", 0.0)),
                    data.get("z", None),
                    data.get("theta", None),
                )

        @self.socketio.on("set_initial_positions")
        def on_set_initial_positions(data):
            with self.msg.lock:
                self.msg.spawn_points = data if isinstance(data, dict) else {}

        @self.socketio.on("set_robot_velocity")
        def on_set_robot_velocity(data):
            name = str(data.get("name", ""))
            vx = float(data.get("vx", 0.0))
            vy = float(data.get("vy", 0.0))
            wz = float(data.get("wz", 0.0))
            with self.msg.lock:
                if self.msg.velocity_cmds is None:
                    self.msg.velocity_cmds = []
                self.msg.velocity_cmds.append((name, vx, vy, wz))

    def start(self, port: int = 5811):
        t = threading.Thread(
            target=lambda: self.socketio.run(
                self.app,
                host="0.0.0.0",
                port=port,
                use_reloader=False,
                debug=False,
                allow_unsafe_werkzeug=True,
            ),
            daemon=True,
        )
        t.start()

    def poll_commands(self) -> WebMsgBuffer:
        with self.msg.lock:
            out = WebMsgBuffer(
                reset_env=self.msg.reset_env,
                restart_match=self.msg.restart_match,
                viewer_point=self.msg.viewer_point,
                viewer_look_at=self.msg.viewer_look_at,
                camera_preset=self.msg.camera_preset,
                teleport_cmd=self.msg.teleport_cmd,
                spawn_points=self.msg.spawn_points,
                velocity_cmds=list(self.msg.velocity_cmds) if self.msg.velocity_cmds is not None else None,
            )
            self.msg.reset_env = False
            self.msg.restart_match = False
            self.msg.viewer_point = None
            self.msg.viewer_look_at = None
            self.msg.camera_preset = None
            self.msg.teleport_cmd = None
            self.msg.spawn_points = None
            self.msg.velocity_cmds = None
            return out

    def emit_frame(self, rgb: np.ndarray):
        image = Image.fromarray(rgb)
        bio = BytesIO()
        image.save(bio, format="JPEG", quality=92)
        payload = base64.b64encode(bio.getvalue()).decode("utf-8")
        self.socketio.emit("new_frame", {"image": payload})

    def emit_robot_states(self, states: dict):
        self.socketio.emit("robot_states", states)

    def set_field_meta(self, field_meta: dict):
        self._field_meta = dict(field_meta)
