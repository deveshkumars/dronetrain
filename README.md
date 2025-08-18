## DroneTrain

### Prerequisites
- Create required directories and fetch MuJoCo Menagerie models:
```bash
mkdir -p plots gifs models
git clone https://github.com/google-deepmind/mujoco_menagerie
```
- Recommended environment setup:
```bash
pip install uv
uv sync
source .venv/bin/activate
```
- Optional system deps (headless rendering/video on Linux):
  - ffmpeg available on PATH (e.g., module load ffmpeg)

### Running via CLI (preferred)
- Train only:
```bash
python main.py --train --env simple --model_xml /absolute/path/to/scene_mjx.xml
```
- Evaluate with saved params:
```bash
python main.py --eval --env simple --steps 200 --video gifs/simple_train.mp4 --model_xml /absolute/path/to/scene_mjx.xml
```
- Train then evaluate in one command:
```bash
python main.py --train --eval --env simple
```

CLI flags:
- `--env`: Brax environment name (we register `simple`).
- `--model_xml`: Optional MuJoCo XML path override for the environment.
- `--model_dir`: Where policy params are saved/loaded (default `models/mjx_brax_policy`).
- `--steps`: Evaluation rollout steps (default 200).
- `--video`: Output video path (set empty to skip).

### Programmatic usage
```python
from dtcore import trainer

# Train
make_inference_fn, params, _ = trainer.train(
    env_name='simple',
    env_kwargs={'model_xml_path': '/absolute/path/to/scene_mjx.xml'},
    model_dir='models/mjx_brax_policy',
)

# Evaluate
trainer.evaluate(
    env_name='simple',
    make_inference_fn=make_inference_fn,
    params=params,
    n_steps=200,
    video_path='gifs/simple_train.mp4',
    env_kwargs={'model_xml_path': '/absolute/path/to/scene_mjx.xml'},
)
```

### Legacy entrypoint
- A thin wrapper remains for quick training:
```bash
python train.py
```

### Notes
- The trainer configures `MUJOCO_GL=egl` and disables JAX traceback filtering for clearer errors.
- Update submodules if you use the included models/utilities:
```bash
git submodule update --init --recursive
# Later updates:
git submodule update --remote --merge submodules/Custom-Crazyflie-Mujoco-Model
```
