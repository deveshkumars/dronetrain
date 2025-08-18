

from datetime import datetime
from etils import epath
import functools
from typing import Any, Dict, Sequence, Tuple, Union
import os
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
from ml_collections import config_dict
os.environ['MUJOCO_GL'] = 'egl'  # or try 'osmesa' if egl doesn't work
# os.environ['JAX_TRACEBACK_FILTERING'] = 'off'  # Full tracebacks
# os.environ['JAX_DEBUG_NANS'] = 'True'  # Raise errors on NaNs
# os.environ['JAX_ENABLE_X64'] = 'True'  # Use double precision
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'  # More detailed XLA info
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Force CPU execution

import jax
from jax import numpy as jnp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from orbax import checkpoint as ocp

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

# tflite conversion code
from orbax.export import ExportManager
from orbax.export import JaxModule
from orbax.export import ServingConfig
import tensorflow as tf

from dtcore import trainer


def main():
  # Simple passthrough to the library API
  print('training...')
  make_inference_fn, params, _ = trainer.train(env_name='simple')
  print('done training')


if __name__ == '__main__':
  main()

