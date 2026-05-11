# MuJoCo PI+ Simulation (Standalone Repo)

This repository is an extracted standalone simulation pipeline for **MuJoCo + `pi_plus`** only.

## Scope

Included:
- MuJoCo multi-robot simulation (`pi_plus`)
- Runtime runner (`mujoco_pi_plus_sim.runner`)
- Simulation manager API/UI (`mujoco_pi_plus_sim.sim_manager`)
- Web visualization pages (`web/`)
- External gait/control tools (`tools/`)

Removed:
- Isaac compatibility library and bridge dependencies
- All `k1` assets/policies

## Repository Layout

- `src/mujoco_pi_plus_sim/`: runtime config, simulator core, manager backend
- `assets/`: `pi_plus` robot, environment, and policy
- `web/`: simulation webview + manager frontend
- `tools/`: gait/control helper scripts

## Environment Setup (uv)

```bash
cd ./mujoco-pi-plus-sim
uv sync
```

Quick dependency check:

```bash
uv run python -c "import mujoco, torch, zmq, flask, fastapi, uvicorn; print('ok')"
```

## Runtime Model (Current)

Current control contract is:
- External control interface: **50Hz** (`dt=0.02`)
- Internal physics integration: **2ms** step (`sim_dt=0.002`)
- Internal substeps per external control tick: **10** (`control_decimation=10`)
- Actuation input to MuJoCo: **joint-based** (`joint_actions` / `joint_targets`)

`cmd_vel` is **not** sent to MuJoCo as direct actuation. It is used by `gait_node` to condition policy output.

## Run Simulation

```bash
uv run mos-sim-run --control-mode joint_target --robot-type pi_plus --team-size 1 --port 5555
```

On macOS:

```bash
uv run mos-sim-run --control-mode joint_target --mujoco-gl cgl --robot-type pi_plus --team-size 1 --port 5555
```

## Run External Gait Pipeline (No Mux)

1. Start simulation:

```bash
uv run mos-sim-run --control-mode joint_target --robot-type pi_plus --team-size 1 --port 5555
```

2. Start runtime velocity publisher (optional but recommended):

```bash
uv run python tools/cmd_vel_node.py --bind tcp://127.0.0.1:6003 --vx 1.0 --vy 0.0 --wz 0.0 --rate 20
```

3. Start gait node (single writer to sim):

```bash
uv run python tools/gait_node.py \
  --sim-host 127.0.0.1 --sim-port 5555 \
  --sensor-bind tcp://127.0.0.1:6001 \
  --intent-bind tcp://127.0.0.1:6002 \
  --cmd-connect tcp://127.0.0.1:6003 \
  --rids 0,7 \
  --policy-device cpu \
  --dt 0.02
```

Notes:
- `gait_node.py` directly communicates with MuJoCo ZMQ (no `control_mux.py`).
- `--dt 0.02` keeps external policy/control at 50Hz.
- MuJoCo internally applies 10 physics substeps per control message.

## ZMQ Protocol (Sim)

Request (external controller -> sim):

```json
{
  "timestamp": 1715420000.123,
  "source": "gait_node",
  "commands": [{"id": 0, "cmd": [1.0, 0.0, 0.0]}],
  "joint_actions": [{"id": 0, "a": [0.0, 0.0, 0.0]}]
}
```

Response (sim -> external controller):
- `state`: world summary
- `sensors`: per-robot sensors including
  - `obs`
  - `joint_pos`, `joint_vel`, `joint_pos_target`
  - `base_pos`, `base_quat_wxyz`
- `control_mode`, `sim_timestamp`, `step_latency`, `ack_timestamp`

## Simulation Manager

```bash
uv run mos-sim-manager --host 0.0.0.0 --port 8000
```

Manager pages:
- UI: `http://127.0.0.1:8000/`
- API docs page: `http://127.0.0.1:8000/manager/docs`
- Swagger: `http://127.0.0.1:8000/docs`

## Notice

This repository is extracted from the original `mos-sim` project and keeps MOS-Sim contributor attribution in source headers.

## Contributors (from mos-sim commit history)

Deduplicated by email:

1. Shibo Xia (`sbxia`) `<xsb25@mails.tsinghua.edu.cn>`
2. 罗绍殷 (`luo-sy24`) `<luo-sy24@mails.tsinghua.edu.cn>`
3. wangju (aka `infrontlight`) `<j-wang24@mails.tsinghua.edu.cn>`, `<1051330335@qq.com>`
4. wegg111 `<1047950878@qq.com>`
