# MuJoCo PI+ Simulation

This repository contains a standalone MuJoCo simulation pipeline for `pi_plus`.

## Scope

Included:
- MuJoCo multi-robot simulation for `pi_plus`
- Runtime runner `mujoco_pi_plus_sim.runner`
- Simulation manager API and UI `mujoco_pi_plus_sim.sim_manager`
- Web visualization pages in `web/`
- Control tools in `tools/`

Removed:
- Isaac compatibility library and bridge dependencies
- All `k1` assets/policies

## Repository Layout

- `src/mujoco_pi_plus_sim/`: runtime config, simulator core, manager backend
- `assets/`: `pi_plus` robot, environment, and policy
- `web/`: simulation webview + manager frontend
- `tools/`: gait/control helper scripts

## Environment Setup

```bash
cd ./mujoco-pi-plus-sim
uv sync
```

Quick dependency check:

```bash
uv run python -c "import mujoco, torch, zmq, flask, fastapi, uvicorn; print('ok')"
```

## MuJoCo Input and Output

Current control contract:
- External control runs at 50Hz
- Internal physics step is 2ms
- MuJoCo runs 10 internal substeps for each external control tick
- Actuation is joint-based through `joint_actions` or `joint_targets`

`cmd_vel` is **not** a direct MuJoCo actuation input.

### ZMQ Input

Request example:

```json
{
  "timestamp": 1715420000.123,
  "source": "controller",
  "commands": [{"id": 0, "cmd": [1.0, 0.0, 0.0]}],
  "joint_actions": [{"id": 0, "a": [0.0, 0.0, 0.0]}]
}
```

Input fields:
- `commands`: optional high-level command per robot, used to fill command-related observation terms
- `joint_actions`: main joint-space actuation input
- `joint_targets` / `joint_pos`: optional joint-position style inputs

### ZMQ Output

Response includes:
- `state`: world summary
- `sensors`: per-robot sensor payload
  - `obs`
  - `joint_pos`, `joint_vel`, `joint_pos_target`
  - `base_pos`, `base_quat_wxyz`
- `control_mode`
- `sim_timestamp`
- `step_latency`
- `ack_timestamp`

## Joint Name and ID Mapping

`joint_actions[*].a` uses the policy joint order. Index table:

| ID | Joint Name |
|---:|---|
| 0 | l_hip_pitch_joint |
| 1 | l_shoulder_pitch_joint |
| 2 | r_hip_pitch_joint |
| 3 | r_shoulder_pitch_joint |
| 4 | l_hip_roll_joint |
| 5 | l_shoulder_roll_joint |
| 6 | r_hip_roll_joint |
| 7 | r_shoulder_roll_joint |
| 8 | l_thigh_joint |
| 9 | l_upper_arm_joint |
| 10 | r_thigh_joint |
| 11 | r_upper_arm_joint |
| 12 | l_calf_joint |
| 13 | l_elbow_joint |
| 14 | r_calf_joint |
| 15 | r_elbow_joint |
| 16 | l_ankle_pitch_joint |
| 17 | r_ankle_pitch_joint |
| 18 | l_ankle_roll_joint |
| 19 | r_ankle_roll_joint |

MuJoCo internal joint vector order is different:

| MuJoCo ID | Joint Name |
|---:|---|
| 0 | l_hip_pitch_joint |
| 1 | l_hip_roll_joint |
| 2 | l_thigh_joint |
| 3 | l_calf_joint |
| 4 | l_ankle_pitch_joint |
| 5 | l_ankle_roll_joint |
| 6 | l_shoulder_pitch_joint |
| 7 | l_shoulder_roll_joint |
| 8 | l_upper_arm_joint |
| 9 | l_elbow_joint |
| 10 | r_hip_pitch_joint |
| 11 | r_hip_roll_joint |
| 12 | r_thigh_joint |
| 13 | r_calf_joint |
| 14 | r_ankle_pitch_joint |
| 15 | r_ankle_roll_joint |
| 16 | r_shoulder_pitch_joint |
| 17 | r_shoulder_roll_joint |
| 18 | r_upper_arm_joint |
| 19 | r_elbow_joint |

Mapping arrays used in code:
- policy -> MuJoCo: `[0, 4, 8, 12, 16, 18, 1, 5, 9, 13, 2, 6, 10, 14, 17, 19, 3, 7, 11, 15]`
- MuJoCo -> policy: `[0, 6, 10, 16, 1, 7, 11, 17, 2, 8, 12, 18, 3, 9, 13, 19, 4, 14, 5, 15]`

## Run Simulation

```bash
uv run mos-sim-run --control-mode joint_target --robot-type pi_plus --team-size 1 --port 5555
```

On macOS:

```bash
uv run mos-sim-run --control-mode joint_target --mujoco-gl cgl --robot-type pi_plus --team-size 1 --port 5555
```

## Example: External Gait Validation Pipeline

1. Start simulation:

```bash
uv run mos-sim-run --control-mode joint_target --robot-type pi_plus --team-size 1 --port 5555
```

2. Start runtime velocity publisher:

```bash
uv run python tools/cmd_vel_node.py --bind tcp://127.0.0.1:6003 --vx 1.0 --vy 0.0 --wz 0.0 --rate 20
```

3. Start gait node:

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
- `gait_node.py` writes directly to MuJoCo ZMQ. `control_mux.py` is not required.
- `--dt 0.02` keeps external policy control at 50Hz.
- MuJoCo internally applies 10 physics substeps per control message.

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

## Contributors from mos-sim commit history

Deduplicated by email:

1. Shibo Xia `sbxia` `<xsb25@mails.tsinghua.edu.cn>`
2. 罗绍殷 `luo-sy24` `<luo-sy24@mails.tsinghua.edu.cn>`
3. wangju / `infrontlight` `<j-wang24@mails.tsinghua.edu.cn>`, `<1051330335@qq.com>`
4. wegg111 `<1047950878@qq.com>`
