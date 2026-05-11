# MuJoCo PI+ Simulation (Standalone Repo)

This repository is an extracted standalone simulation pipeline for **MuJoCo + `pi_plus`** only.

## Scope

Included:
- MuJoCo multi-robot simulation pipeline (`pi_plus`)
- Simulation runner (`sim2sim_runner.py`)
- Simulation manager API/UI (`sim_manager.py`, `app/sim_manager.py`)
- Web visualization and control pages (`web/`)

Removed:
- Isaac-related vendored compatibility library (`vendor/sim2simlib`)
- All `k1` robot assets and policies
- Monorepo forwarding dependency to `simulation.labbridge`

## Repository Layout

- `sim2sim_runner.py`: main simulation entry
- `sim_manager.py`: manager entrypoint
- `app/`: runtime config, simulator core, manager backend
- `assets/`: `pi_plus` robot, environment, and `pi_plus` policy
- `web/`: simulation webview + manager frontend
- `tools/`: utility scripts

## Environment Setup

```bash
cd ./mujoco-pi-plus-sim
conda create -n mujoco312 python=3.12 -y
conda activate mujoco312
pip install -r requirements.txt
```

Quick dependency check:

```bash
python -c "import mujoco, torch, zmq, flask, fastapi, uvicorn; print('ok')"
```

## Run Simulation (pi_plus)

```bash
cd ./mujoco-pi-plus-sim
python sim2sim_runner.py --robot-type pi_plus --team-size 3
```

Notes:
- `--robot-type` is `pi_plus` only in this repo.
- `--team-size` range: `0..7`
- default webview: `http://localhost:5811`
- default ZMQ REP endpoint: `tcp://*:5555`

## Run Simulation Manager

Start manager:

```bash
cd ./mujoco-pi-plus-sim
python sim_manager.py --host 0.0.0.0 --port 8000
```

Manager pages:
- UI: `http://127.0.0.1:8000/`
- API docs page: `http://127.0.0.1:8000/manager/docs`
- Swagger: `http://127.0.0.1:8000/docs`

## Optional: Run Manager with Uvicorn

```bash
cd ./mujoco-pi-plus-sim
uvicorn app.sim_manager:app --host 0.0.0.0 --port 8000
```

## Common Runtime Options

Simulation:
- `--team-size <0..7>`
- `--port <zmq_port>`
- `--webview-port <port>`
- `--web-fps <int>`
- `--web-width <int>`
- `--web-height <int>`
- `--policy-device cpu|gpu`
- `--use-referee` / `--no-use-referee`

Manager start request supports:
- `team_size`
- `zmq_port`
- `webview_port`
- `policy_device`
- `use_referee`

## ZMQ Command Format

Single command request:

```json
{"cmd":[vx,vy,w], "id":0, "timestamp": 0, "source":"xxx"}
```

## Notes

- This repo is intentionally narrowed to `pi_plus` pipeline maintenance and deployment.
- If you need multi-backend simulation (Isaac/Genesis/bridge), use the original monorepo instead.

## Use With uv

Install dependencies and create lockfile:

```bash
cd ./mujoco-pi-plus-sim
uv sync
```

Run simulation:

```bash
uv run mos-sim-run --robot-type pi_plus --team-size 3
```

Run manager:

```bash
uv run mos-sim-manager --host 0.0.0.0 --port 8000
```

## Notice

This repository is extracted from the original `mos-sim` project and keeps the original MOS-Sim contributor attribution in source headers.

## Contributors (from mos-sim commit history)

Deduplicated by email:

1. Shibo Xia (`sbxia`) `<xsb25@mails.tsinghua.edu.cn>`
2. 罗绍殷 (`luo-sy24`) `<luo-sy24@mails.tsinghua.edu.cn>`
3. wangju `<j-wang24@mails.tsinghua.edu.cn>`
4. infrontlight `<1051330335@qq.com>`
5. wegg111 `<1047950878@qq.com>`
