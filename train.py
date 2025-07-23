

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
# import jax.numpy as jnp



class SimpleEnv(PipelineEnv):
  """Crazyflie 2.0 to 1 m and keep it stable."""

  def __init__(self,
               target_height: float = 1.0,
               height_reward_weight: float = 5.0,
               ctrl_cost_weight: float = 0.1,
               stability_cost_weight: float = 0.5,
               reset_noise_scale: float = 1e-2,
               reward_distance_scale: float = 1.2,
               reward_effort_weight: float = 0.05,
               reward_action_smoothness_weight: float = 0.05,
               **kwargs):
    # load your MJX model
    mj_model = mujoco.MjModel.from_xml_path("/users/dkumar23/scratch/dronetrain/mujoco_menagerie/bitcraze_crazyflie_2/scene_mjx.xml")   # or full path
    mj_model.opt.solver       = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.iterations   = 10
    mj_model.opt.ls_iterations= 10

    sys = mjcf.load_model(mj_model)

    # number of physics steps per control action
    physics_steps = 5
    kwargs['n_frames'] = kwargs.get('n_frames', physics_steps)
    kwargs['backend']  = 'mjx'

    super().__init__(sys, **kwargs)

    self.target_height          = target_height
    self.height_reward_weight   = height_reward_weight
    self.ctrl_cost_weight       = ctrl_cost_weight
    self.stability_cost_weight  = stability_cost_weight
    self.reset_noise_scale      = reset_noise_scale
    self.target_pos = jnp.array([0.0, 0.0, 1.0])
    self.reward_distance_scale = reward_distance_scale
    self.reward_effort_weight = reward_effort_weight
    self.reward_action_smoothness_weight = reward_action_smoothness_weight

    # cache body index for the payload box

  def reset(self, rng: jnp.ndarray) -> State:
    rng, rng1, rng2 = jax.random.split(rng, 3)
    # add a little noise around default qpos0 / qvel0
    low, hi = -self.reset_noise_scale, self.reset_noise_scale
    qpos = self.sys.qpos0# + jax.random.uniform(rng1, (self.sys.nq,), low, hi)
    qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=-0.01, maxval=0.01)
    data = self.pipeline_init(qpos, qvel)

    # initial observation, zero reward/done/metrics
    obs    = self._get_obs(data, jnp.zeros(self.sys.nu))
    reward = jnp.array(0.0)
    done   = jnp.array(0.0)
    metrics = {
      'x'       : data.qpos[0],
      'y'       : data.qpos[1],
      'z'       : data.qpos[2],
    }
    return State(data, obs, reward, done, metrics)

  def step(self, state: State, action: jnp.ndarray) -> State:
    data0 = state.pipeline_state
    data1 = self.pipeline_step(data0, action)

    # 1) height reward: encourage box_z ≈ target_height
    xyz = data1.qpos[0:3]
    xyz_err = jnp.linalg.norm(xyz - self.target_pos)  


    reward_pos = 1.0 / (1.0 + self.reward_distance_scale * jnp.square(xyz_err))

    #uprightness reward from quaternion
    # quat = data1.xquat[1] #drone quaternion
    # reward_up = jnp.square(quat[2]) #z component of quaternion
    # reward_up = jnp.square((reward_up + 1.0) / 2.0)
    # body 1 is the quadcopter
    # xmat[1] is row-major 3×3; the third column is the body z-axis in world frame
    body_z_world = data1.xmat[1, 2::3]        # shape (3,) → (xx, yx, zx) etc.
    upright_cos  = body_z_world[2]            # dot([0,0,1], body_z_world)

    reward_up = (upright_cos + 1.0) / 2.0     # 1 when upright, 0 when upside-down

    #spin reward
    root_ang_vel_z = data1.qvel[5]           # assuming free-joint: ang vel (wx, wy, wz)
    spinnage   = jnp.square(root_ang_vel_z)
    reward_spin = 1.0 / (1.0 + jnp.square(spinnage))

    #effort reward
    effort      = jnp.sum(jnp.square(action))
    reward_effort = self.reward_effort_weight * jnp.exp(-effort)

    reward = reward_pos + reward_pos*(reward_spin) + reward_effort

    # 3) stability cost: penalize box angular velocity (rough proxy)
    #    we can get body angular vel from data1.cvel (twist) for each body:

    obs = self._get_obs(data1, action)
    # update metrics for logging
    state.metrics.update(
      x=xyz[0],
      y=xyz[1],
      z=xyz[2],
    )

    #done = jnp.where((height <= 0.0) | (height >= 2.0), 1.0, 0.0)
    # --- done (termination) signal ---------------------------------------------
    # contact with the floor?
    hard_contact = data1._impl.ncon > 0           # 1 if any contact pair is active

    # body orientation: roll / pitch must stay within ±45 deg
        # body orientation: roll / pitch must stay within ±45 deg
    root_quat = data1.xquat[1]              # body 1 is the quad itself (world=0)
    roll, pitch, _ = math.quat_to_euler(root_quat)
    flipped = (jnp.abs(roll) > jnp.pi/4) | (jnp.abs(pitch) > jnp.pi/4)

    # done if  height is too low
    too_low = xyz[2] < 0.05                  # boolean

    done = jnp.where(flipped | too_low, 1.0, 0.0) 
    #done = jnp.array(0.0)
    print("step done")
    return state.replace(pipeline_state=data1,
                         obs=obs,
                         reward=reward,
                         done=done)

  def _get_obs(self, data, action):
    # you can observe:
    #  - box xpos & orientation  (7 dims)
    #  - cf2_i qpos & qvel for each drone
    #  - cable stretch (if desired)
    #  - last action
    obs = jnp.concatenate([data.qpos,
                           data.qvel,
                           action])
    return obs

cam = mujoco.MjvCamera()

mujoco.mjv_defaultCamera(cam)

# Option 1: Isometric view (recommended for general viewing)
cam.lookat = [0, 0, 0.1]    # Look at where the Crazyflie is (z=0.1)
cam.distance = 3          # Much closer than 10
cam.elevation = -30         # Look down at an angle
cam.azimuth = 45   

envs.register_environment('simple', SimpleEnv)

# instantiate the environment
env_name = 'simple'
env = envs.get_environment(env_name)

print("# actuators (sys.nu):", env.sys.nu)   # should print 12
print("action_shape:", env.action_size)  
print("position:", env.sys.qpos0)  

# define the jit reset/step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# initialize the state
state = jit_reset(jax.random.PRNGKey(0))
rollout = [state.pipeline_state]

# # grab a trajectory
# for i in range(100):
#   #ctrl = 0.075 * jnp.ones(env.sys.nu)
#   ctrl = jnp.array([0.075, 0.070, 0.070, 0.075])
#   state = jit_step(state, ctrl)
#   rollout.append(state.pipeline_state)

# media.write_video(path="gifs/train_low.mp4", images=env.render(rollout, camera=cam), fps=1.0 / env.dt)


env_name = 'simple'
env    = envs.get_environment(env_name)

train_fn = functools.partial(
    ppo.train,
    num_timesteps=1, # default 1000000
    num_evals=5,
    reward_scaling=0.1,
    episode_length=200,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=1, # default 10
    num_minibatches=1, # default 24
    num_updates_per_batch=1, # default 8
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-3,
    num_envs=2, # default 512
    batch_size=2, # default 512
    seed=0,
)

y_data = []
ydataerr = []
times = [datetime.now()]

xdata, ydata, ydataerr = [], [], []
times = [datetime.now()]

def progress(num_steps, metrics):
  times.append(datetime.now())
  print(num_steps, metrics['eval/episode_reward'])
  print('position:', metrics['eval/episode_x'], metrics['eval/episode_y'], metrics['eval/episode_z'])
  xdata.append(num_steps)
  ydata.append(metrics['eval/episode_reward'])
  ydataerr.append(metrics['eval/episode_reward_std'])
  times.append(datetime.now())
 




print('training...')
make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)


#@title Save Model
model_path = 'models/mjx_brax_policy'
model.save_params(model_path, params)

#@title Load Model and Define Inference Function
params = model.load_params(model_path)

inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)


eval_env = envs.get_environment(env_name)

jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

# initialize the stateS
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
rollout = [state.pipeline_state]

# grab a trajectory
n_steps = 200
render_every = 2

for i in range(n_steps):
  act_rng, rng = jax.random.split(rng)
  ctrl, _ = jit_inference_fn(state.obs, act_rng)
  state = jit_step(state, ctrl)
  rollout.append(state.pipeline_state)



print(len(rollout))

print("writing video")
media.write_video(path="gifs/simple_train.mp4", images=env.render(rollout, camera=cam), fps=1.0 / env.dt)
print('done training')
plt.plot(xdata, ydata, label='Episode Reward')
plt.xlabel('Number of Steps')
plt.ylabel('Episode Reward')
plt.title('Episode Reward vs. Number of Steps')
plt.legend()
plt.savefig('plots/train.png')


# ------------------------------------------------------------------------------------------------
# Step Two: turn the model into a TF lite
# ------------------------------------------------------------------------------------------------


# Create the network and inference function
# env = envs.get_pretrained('simple') # env is already defined above

def identity_preprocess(observation, preprocessor_params):
  return observation

ppo_network = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size
)

# Create policy module and apply function

policy_module = ppo_network.policy_network

def apply_fn(params, obs):
  # Get action distribution params from policy network
  # The policy module expects only one params argument, not two
  action_mean = policy_module.apply(params, obs, train=False)
  # Sample deterministically (take mean action)
  return action_mean

jax_module = JaxModule(params, apply_fn)


# This code snippet converts a JAX model to TFLite through TF SavedModel.
from orbax.export import ExportManager
from orbax.export import JaxModule
from orbax.export import ServingConfig
import tensorflow as tf
import jax.numpy as jnp

# default_method_key = list(jax_module.methods.keys())[0]


# # Option 1: Simply save the model via `tf.saved_model.save` if no need for pre/post
# # processing.
default_key = list(jax_module.methods.keys())[0]
tf.saved_model.save(
    jax_module,
    '/models/',
    signatures=jax_module.methods[default_key].get_concrete_function(
        tf.TensorSpec(shape=(None,), dtype=tf.float32, name="input")
    ),
    options=tf.saved_model.SaveOptions(experimental_custom_gradients=True),
)
converter = tf.lite.TFLiteConverter.from_saved_model('/models/')
tflite_model = converter.convert()

# # Option 2: Define pre/post processing TF functions (e.g. (de)?tokenize).
# serving_config = ServingConfig(
#     'Serving_default',
#     # Corresponds to the input signature of `tf_preprocessor`
#     input_signature=[tf.TensorSpec(shape=(env.observation_size,), dtype=tf.float32, name='input')],
#     tf_preprocessor=lambda x: x,
#     tf_postprocessor=lambda out: {'output': out}
# )
# export_mgr = ExportManager(jax_module, [serving_config])
# export_mgr.save('/models/')
# print("saved model")
# converter = tf.lite.TFLiteConverter.from_saved_model('/models/')
# tflite_model = converter.convert()

# # Option 3: Convert from TF concrete function directly
# converter = tf.lite.TFLiteConverter.from_concrete_functions(
#     [
#         jax_module.methods[JaxModule.DEFAULT_METHOD_KEY].get_concrete_function(
#             tf.TensorSpec(shape=(None,), dtype=tf.float32, name="input")
#         )
#     ]
# )
# tflite_model = converter.convert()


# Run the model with LiteRT
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create sample input data
input_data = np.zeros((env.observation_size,), dtype=np.float32)
interpreter.set_tensor(input_details[0]["index"], input_data)
interpreter.invoke()
result = interpreter.get_tensor(output_details[0]["index"])
print("TFLite model output:", result)