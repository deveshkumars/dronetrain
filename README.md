
## Double Check
if these directories don't exist, make them below: mujoco_managerie, plots, gifs, models

```
mkdir plots
mkdir gifs
mkdir models 
git clone https://github.com/google-deepmind/mujoco_menagerie
```

# How to run
```
pip install uv
source .venv/bin/activate
uv sync
uv run single_train.py
```


pip install tf-nightly --upgrade
pip install jax --upgrade
pip install orbax-export --upgrade

make sure to run module load ffmpeg 
^ for RHEL servers