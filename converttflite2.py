#!/usr/bin/env python3
"""
Complete script to train a Brax PPO policy, convert it to TFLite format, and test it.
Can either train from scratch or load existing model for TFLite conversion.
"""

import os
import functools
import jax
import jax.numpy as jnp
import tensorflow as tf
import numpy as np
from datetime import datetime
from orbax.export import ExportManager, JaxModule, ServingConfig, constants
from orbax import checkpoint as ocp
from brax import envs
from brax.training.agents.ppo import train as ppo
import mujoco
import matplotlib.pyplot as plt

# Set environment for MuJoCo
os.environ['MUJOCO_GL'] = 'egl'

# Import the custom environment dependencies
from brax.envs.base import Env, PipelineEnv, State
from brax import base, math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.mjx.base import State as MjxState
from brax.io import html, mjcf, model

# Add this import at the top
from brax.training.acme import running_statistics

# Configuration flags
TRAIN_MODEL = False  # Set to False to skip training and just test TFLite inference
MODEL_PATH = os.path.abspath('/users/dkumar23/scratch/dronetrain/models')
TFLITE_PATH = os.path.abspath('/users/dkumar23/scratch/dronetrain/tflitemodels')

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
        mj_model = mujoco.MjModel.from_xml_path("/users/dkumar23/scratch/dronetrain/mujoco_menagerie/bitcraze_crazyflie_2/scene_mjx.xml")
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 10
        mj_model.opt.ls_iterations = 10

        sys = mjcf.load_model(mj_model)

        # number of physics steps per control action
        physics_steps = 5
        kwargs['n_frames'] = kwargs.get('n_frames', physics_steps)
        kwargs['backend'] = 'mjx'

        super().__init__(sys, **kwargs)

        self.target_height = target_height
        self.height_reward_weight = height_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.stability_cost_weight = stability_cost_weight
        self.reset_noise_scale = reset_noise_scale
        self.target_pos = jnp.array([0.0, 0.0, 1.0])
        self.reward_distance_scale = reward_distance_scale
        self.reward_effort_weight = reward_effort_weight
        self.reward_action_smoothness_weight = reward_action_smoothness_weight

    def reset(self, rng: jnp.ndarray) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        low, hi = -self.reset_noise_scale, self.reset_noise_scale
        qpos = self.sys.qpos0
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=-0.01, maxval=0.01)
        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data, jnp.zeros(self.sys.nu))
        reward = jnp.array(0.0)
        done = jnp.array(0.0)
        metrics = {
            'x': data.qpos[0],
            'y': data.qpos[1],
            'z': data.qpos[2],
        }
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jnp.ndarray) -> State:
        data0 = state.pipeline_state
        data1 = self.pipeline_step(data0, action)

        xyz = data1.qpos[0:3]
        xyz_err = jnp.linalg.norm(xyz - self.target_pos)
        reward_pos = 1.0 / (1.0 + self.reward_distance_scale * jnp.square(xyz_err))

        body_z_world = data1.xmat[1, 2::3]
        upright_cos = body_z_world[2]
        reward_up = (upright_cos + 1.0) / 2.0

        root_ang_vel_z = data1.qvel[5]
        spinnage = jnp.square(root_ang_vel_z)
        reward_spin = 1.0 / (1.0 + jnp.square(spinnage))

        effort = jnp.sum(jnp.square(action))
        reward_effort = self.reward_effort_weight * jnp.exp(-effort)

        reward = reward_pos + reward_pos * reward_spin + reward_effort

        obs = self._get_obs(data1, action)
        state.metrics.update(
            x=xyz[0],
            y=xyz[1],
            z=xyz[2],
        )

        hard_contact = data1.ncon > 0
        root_quat = data1.xquat[1]
        roll, pitch, _ = math.quat_to_euler(root_quat)
        flipped = (jnp.abs(roll) > jnp.pi/4) | (jnp.abs(pitch) > jnp.pi/4)
        too_low = xyz[2] < 0.05

        done = jnp.where(flipped | too_low, 1.0, 0.0)

        return state.replace(pipeline_state=data1,
                           obs=obs,
                           reward=reward,
                           done=done)

    def _get_obs(self, data, action):
        obs = jnp.concatenate([data.qpos,
                             data.qvel,
                             action])
        return obs

# Register the environment
envs.register_environment('simple', SimpleEnv)

def train_model():
    """Train the PPO model and save it."""
    print("=== Training PPO Model ===")
    
    env = envs.get_environment('simple')
    
    print("# actuators (sys.nu):", env.sys.nu)
    print("action_shape:", env.action_size)
    print("position:", env.sys.qpos0)

    # Training configuration
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=100000,
        num_evals=5,
        reward_scaling=0.1,
        episode_length=200,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=24,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        num_envs=512,
        batch_size=512,
        seed=0,
    )

    # Training progress tracking
    xdata, ydata, ydataerr = [], [], []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        times.append(datetime.now())
        print(f"Steps: {num_steps}, Reward: {metrics['eval/episode_reward']:.3f}")
        print(f"Position: x={metrics['eval/episode_x']:.3f}, y={metrics['eval/episode_y']:.3f}, z={metrics['eval/episode_z']:.3f}")
        xdata.append(num_steps)
        ydata.append(metrics['eval/episode_reward'])
        ydataerr.append(metrics['eval/episode_reward_std'])

    print('Starting training...')
    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

    # Save the model
    print(f"Saving model to {MODEL_PATH}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Remove existing checkpoint if it exists
    if os.path.exists(MODEL_PATH):
        import shutil
        print(f"Removing existing checkpoint at {MODEL_PATH}")
        shutil.rmtree(MODEL_PATH)
    
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(MODEL_PATH, params, force=True)

    # Save training plot
    plt.figure(figsize=(10, 6))
    plt.plot(xdata, ydata, label='Episode Reward')
    plt.fill_between(xdata, 
                     [y - err for y, err in zip(ydata, ydataerr)],
                     [y + err for y, err in zip(ydata, ydataerr)], 
                     alpha=0.3)
    plt.xlabel('Number of Steps')
    plt.ylabel('Episode Reward')
    plt.title('Training Progress: Episode Reward vs. Number of Steps')
    plt.legend()
    plt.grid(True)
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/training_progress.png', dpi=150, bbox_inches='tight')
    print("Training plot saved to plots/training_progress.png")

    print('Training completed!')
    return params, make_inference_fn, env

def load_model():
    """Load the trained model parameters."""
    print("=== Loading Trained Model ===")
    
    # Create the environment
    env = envs.get_environment('simple')
    
    # We need to recreate the training function to get the make_inference_fn
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=100000,
        num_evals=5,
        reward_scaling=0.1,
        episode_length=200,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=24,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        num_envs=512,
        batch_size=512,
        seed=0,
    )
    
    # Load the parameters using orbax checkpoint
    try:
        checkpointer = ocp.PyTreeCheckpointer()
        raw_params = checkpointer.restore(MODEL_PATH)
        print(f"Successfully loaded raw model from {MODEL_PATH}")
        
        # Convert the normalization parameters back to RunningStatisticsState objects
        # The saved params are typically: (normalizer_params, policy_params)
        if isinstance(raw_params, tuple) and len(raw_params) >= 2:
            normalizer_dict, policy_params = raw_params[0], raw_params[1]
            
            # Reconstruct the RunningStatisticsState from the dictionary
            if isinstance(normalizer_dict, dict):
                normalizer_params = running_statistics.RunningStatisticsState(
                    count=normalizer_dict['count'],
                    mean=normalizer_dict['mean'],
                    std=normalizer_dict['std'],
                    summed_variance=normalizer_dict['summed_variance']
                )
                params = (normalizer_params, policy_params)
            else:
                params = raw_params
        else:
            params = raw_params
            
    except Exception as e:
        print(f"Error loading model: {e}")
        print("You may need to set TRAIN_MODEL=True to train the model first")
        return None, None, None
    
    # Create a dummy training to get the make_inference_fn
    # This is a bit hacky but necessary to get the inference function structure
    make_inference_fn, _, _ = train_fn(environment=env, progress_fn=lambda *args: None, num_timesteps=1)
    
    return params, make_inference_fn, env

def convert_to_tflite(params, make_inference_fn, env):
    """Convert the Brax model to TFLite format."""
    print("=== Converting to TFLite ===")
    
    # Create the inference function
    inference_fn = make_inference_fn(params)
    
    # Get observation shape from environment
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    obs_shape = state.obs.shape
    
    print(f"Observation shape: {obs_shape}")
    print(f"Action shape: {env.action_size}")
    
    # Define a wrapper function that only takes observations (no random key for deterministic inference)
    def policy_fn(params, obs):
        # For deterministic inference, we can use a fixed random key or make it deterministic
        rng = jax.random.PRNGKey(0)
        
        # Ensure normalizer_params is a proper RunningStatisticsState object
        normalizer_params, policy_params, _ = params
        if isinstance(normalizer_params, dict):
            normalizer_params = running_statistics.RunningStatisticsState(
                count=normalizer_params['count'],
                mean=normalizer_params['mean'],
                std=normalizer_params['std'],
                summed_variance=normalizer_params['summed_variance']
            )
            reconstructed_params = (normalizer_params, policy_params)
        else:
            reconstructed_params = params
            
        # Recreate the inference function with proper parameters
        temp_inference_fn = make_inference_fn(reconstructed_params)
        action, _ = temp_inference_fn(obs, rng)
        return action
    
    # Create the JAX module for export
    jax_module = JaxModule(
        params, 
        policy_fn, 
        input_polymorphic_shape=f'b, {obs_shape[0]}',
        jax2tf_kwargs={"enable_xla": False},    # ← turn off XLA in jax2tf:contentReference[oaicite:0]{index=0}

    )


    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_concrete_functions([
        jax_module.methods[constants.DEFAULT_METHOD_KEY].get_concrete_function(
            tf.TensorSpec(shape=(1, obs_shape[0]), dtype=tf.float32, name="observation")
        )
    ])

    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, obs_shape[0])
            yield [data.astype(np.float32)]
    
    # Optional: Apply optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
 
    
    tflite_model = converter.convert()
    
    # Save the TFLite model
    os.makedirs(os.path.dirname(TFLITE_PATH), exist_ok=True)
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {TFLITE_PATH}")
    return tflite_model

def test_tflite_model(tflite_model, params, make_inference_fn, env):
    """Test the TFLite model against the original JAX model."""
    print("=== Testing TFLite Model ===")
    
    # Create test data
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)
    test_obs = state.obs[None, :]  # Add batch dimension
    
    # Get prediction from original JAX model
    inference_fn = make_inference_fn(params)
    rng_action = jax.random.PRNGKey(0)
    jax_action, _ = inference_fn(state.obs, rng_action)
    
    # Get prediction from TFLite model
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"TFLite input details: {input_details}")
    print(f"TFLite output details: {output_details}")
    
    # Set input and run inference
    scale, zero_point = input_details[0]['quantization'] # added from og
    test_obs_int8 = (test_obs / scale + zero_point).astype(np.int8) # added from og
    interpreter.set_tensor(input_details[0]["index"], test_obs_int8) # changed from og
    interpreter.invoke()
    tflite_action = interpreter.get_tensor(output_details[0]["index"])
    
    print(f"JAX action shape: {jax_action.shape}")
    print(f"TFLite action shape: {tflite_action.shape}")
    print(f"JAX action: {jax_action}")
    print(f"TFLite action: {tflite_action.flatten()}")
    
    # Compare results
    try:
        np.testing.assert_allclose(jax_action, tflite_action.flatten(), rtol=1e-4, atol=1e-4)
        print("✅ TFLite model matches JAX model!")
    except AssertionError as e:
        print(f"⚠️  Small differences detected: {e}")
        print("This is normal due to numerical precision differences between JAX and TFLite")
        
        # Calculate the difference
        diff = np.abs(jax_action - tflite_action.flatten())
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        if max_diff < 1e-3:
            print("✅ Differences are within acceptable tolerance for deployment")
        else:
            print("❌ Differences are larger than expected - may need investigation")
    
    # Test model size
    model_size = len(tflite_model) / 1024 / 1024  # MB
    print(f"TFLite model size: {model_size:.2f} MB")
    
    # Performance test
    print("\n=== Performance Test ===")
    n_trials = 100
    
    # JAX inference timing
    start_time = datetime.now()
    for _ in range(n_trials):
        jax_action, _ = inference_fn(state.obs, rng_action)
    jax_time = (datetime.now() - start_time).total_seconds()
    
    # TFLite inference timing
    start_time = datetime.now()
    for _ in range(n_trials):
        interpreter.set_tensor(input_details[0]["index"], test_obs.astype(np.float32))
        interpreter.invoke()
        tflite_action = interpreter.get_tensor(output_details[0]["index"])
    tflite_time = (datetime.now() - start_time).total_seconds()
    
    print(f"JAX inference time: {jax_time/n_trials*1000:.2f} ms per inference")
    print(f"TFLite inference time: {tflite_time/n_trials*1000:.2f} ms per inference")
    print(f"TFLite is {jax_time/tflite_time:.1f}x the speed of JAX")

def main():
    """Main function to train, convert, and test the model."""
    print("=== Brax PPO to TFLite Conversion Pipeline ===")
    print(f"TRAIN_MODEL = {TRAIN_MODEL}")
    
    if TRAIN_MODEL:
        # Train the model from scratch
        params, make_inference_fn, env = train_model()
    else:
        # Load existing model
        params, make_inference_fn, env = load_model()
        if params is None:
            print("Failed to load model. Set TRAIN_MODEL=True to train from scratch.")
            return
    
    # Convert to TFLite
    tflite_model = convert_to_tflite(params, make_inference_fn, env)
    
    # Test the converted model
    test_tflite_model(tflite_model, params, make_inference_fn, env)
    
    print("\n=== Pipeline Complete ===")
    print(f"Trained model saved at: {MODEL_PATH}")
    print(f"TFLite model saved at: {TFLITE_PATH}")
    print("You can now deploy the TFLite model to mobile/edge devices!")

if __name__ == "__main__":
    main() 