# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class MLPActor(nn.Module):
    def __init__(self, layer_dims: list[int]):
        super().__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(layer_dims) - 2:
                layers.append(nn.ELU())
        self.actor = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(x)


def _load_checkpoint_compat(path: Path, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except ModuleNotFoundError as e:
        if "rsl_rl" not in str(e):
            raise
        rsl_rl_mod = types.ModuleType("rsl_rl")
        utils_pkg = types.ModuleType("rsl_rl.utils")
        utils_mod = types.ModuleType("rsl_rl.utils.utils")

        class Normalizer:
            pass

        Normalizer.__module__ = "rsl_rl.utils.utils"
        utils_mod.Normalizer = Normalizer
        rsl_rl_mod.utils = utils_pkg
        utils_pkg.utils = utils_mod
        sys.modules.setdefault("rsl_rl", rsl_rl_mod)
        sys.modules.setdefault("rsl_rl.utils", utils_pkg)
        sys.modules.setdefault("rsl_rl.utils.utils", utils_mod)
        return torch.load(path, map_location=map_location, weights_only=False)


@dataclass
class PolicyShape:
    obs_dim: int
    action_dim: int


class PolicyGaitController:
    def __init__(self, policy_path: Path, requested_device: str):
        self.device = self._resolve_device(requested_device)
        self.policy, self.shape = self._load_policy(policy_path)

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        req = str(requested).strip().lower()
        if req == "cpu":
            return torch.device("cpu")
        if req == "gpu":
            if torch.cuda.is_available():
                return torch.device("cuda")
            print("[PolicyGaitController] policy device requested=gpu but CUDA is unavailable, fallback to cpu")
            return torch.device("cpu")
        raise ValueError(f"Unsupported policy device: {requested}")

    def _load_policy(self, policy_path: Path) -> tuple[nn.Module, PolicyShape]:
        ckpt = _load_checkpoint_compat(policy_path, map_location=self.device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Unsupported policy checkpoint format: {type(state_dict)}")
        actor_state = {k: v for k, v in state_dict.items() if k.startswith("actor.")}
        if not actor_state:
            raise RuntimeError("Checkpoint does not contain actor.* weights")

        actor_layer_dims: list[int] = []
        actor_weight_keys = sorted(
            (k for k in actor_state if re.match(r"^actor\.\d+\.weight$", k)),
            key=lambda s: int(s.split(".")[1]),
        )
        for i, wk in enumerate(actor_weight_keys):
            w = actor_state[wk]
            out_dim, in_dim = int(w.shape[0]), int(w.shape[1])
            if i == 0:
                actor_layer_dims.append(in_dim)
            actor_layer_dims.append(out_dim)

        policy = MLPActor(layer_dims=actor_layer_dims).to(self.device)
        policy.load_state_dict(actor_state, strict=True)
        policy.eval()
        shape = PolicyShape(obs_dim=int(actor_layer_dims[0]), action_dim=int(actor_layer_dims[-1]))
        return policy, shape

    def infer_actions(self, obs_batch: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            obs_tensor = torch.from_numpy(obs_batch).to(self.device)
            act_batch = self.policy(obs_tensor).detach().cpu().numpy().astype(np.float32, copy=False)
        if act_batch.ndim == 1:
            act_batch = act_batch.reshape(1, -1)
        return act_batch
