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
python main.py --train --env simple
```
- Evaluate with saved params:
```bash
python main.py --eval --env simple --steps 200 --video gifs/simple_train.mp4
```
- Train then evaluate in one command:
```bash
python main.py --train --eval --env simple
```

CLI flags:
- `--env`: Brax environment name (we register `simple`).
- `--model_xml`: Optional MuJoCo XML path override for the environment (defaults to `submodules/Custom-Crazyflie-Mujoco-Model/scene_mjx.xml`).
- `--model_dir`: Where policy params are saved/loaded (default `models/mjx_brax_policy`).
- `--steps`: Evaluation rollout steps (default 200).
- `--video`: Output video path (set empty to skip).

### Automatic Workflow After Training
When using `--train`, the system automatically performs a complete workflow:

1. **Training**: Trains the policy and saves parameters to `--model_dir`
2. **File Movement**: Moves the saved `.pkl` file to `weights_to_firmware/input_model/`
3. **C Code Generation**: Runs the weights-to-firmware conversion to generate C code
4. **Firmware Integration**: Moves `network_evaluate.c` to `firmware/network_evaluate.h`

This creates a seamless pipeline from training completion to having firmware-ready neural network code.

### Programmatic usage
```python
from dtcore import trainer

# Train (uses default scene_mjx.xml from submodules)
make_inference_fn, params, _ = trainer.train(
    env_name='simple',
    model_dir='models/mjx_brax_policy',
)

# Train with custom XML path
make_inference_fn, params, _ = trainer.train(
    env_name='simple',
    env_kwargs={'model_xml_path': 'custom_scene.xml'},
    model_dir='models/mjx_brax_policy',
)

# Evaluate (uses default scene_mjx.xml from submodules)
trainer.evaluate(
    env_name='simple',
    make_inference_fn=make_inference_fn,
    params=params,
    n_steps=200,
    video_path='gifs/simple_train.mp4',
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

### Project Structure
```
dronetrain/
├── main.py                 # Main entrypoint with automatic workflow
├── firmware/               # Generated firmware files
│   └── network_evaluate.h  # Neural network code for drone firmware
├── weights_to_firmware/    # Submodule for model conversion
│   ├── input_model/        # Input model files (.pkl)
│   └── output_model/       # Generated C code output
└── models/                 # Trained model parameters
```
