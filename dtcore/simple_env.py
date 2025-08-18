from typing import Any
import jax
from jax import numpy as jnp
import mujoco
from brax import envs, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf


class SimpleEnv(PipelineEnv):
    """Crazyflie 2.0 to 1 m and keep it stable."""

    def __init__(
        self,
        target_height: float = 1.0,
        height_reward_weight: float = 5.0,
        ctrl_cost_weight: float = 0.1,
        stability_cost_weight: float = 0.5,
        reset_noise_scale: float = 1e-2,
        reward_distance_scale: float = 1.2,
        reward_effort_weight: float = 0.05,
        reward_action_smoothness_weight: float = 0.05,
        model_xml_path: str = "/users/dkumar23/scratch/dronetrain/mujoco_menagerie/bitcraze_crazyflie_2/scene_mjx.xml",
        physics_steps: int = 5,
        **kwargs: Any,
    ):
        # load your MJX model
        mj_model = mujoco.MjModel.from_xml_path(model_xml_path)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 10
        mj_model.opt.ls_iterations = 10

        sys = mjcf.load_model(mj_model)

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

        reward = reward_pos + reward_pos * (reward_spin) + reward_effort

        obs = self._get_obs(data1, action)
        state.metrics.update(
            x=xyz[0],
            y=xyz[1],
            z=xyz[2],
        )

        root_quat = data1.xquat[1]
        roll, pitch, _ = math.quat_to_euler(root_quat)
        flipped = (jnp.abs(roll) > jnp.pi / 4) | (jnp.abs(pitch) > jnp.pi / 4)
        too_low = xyz[2] < 0.05
        done = jnp.where(flipped | too_low, 1.0, 0.0)

        return state.replace(
            pipeline_state=data1,
            obs=obs,
            reward=reward,
            done=done,
        )

    def _get_obs(self, data, action):
        obs = jnp.concatenate([data.qpos, data.qvel])
        return obs


def make_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat = [0, 0, 0.1]
    cam.distance = 3
    cam.elevation = -30
    cam.azimuth = 45
    return cam


def register_env() -> None:
    """Registers the environment with Brax registry under name 'simple'."""
    envs.register_environment('simple', SimpleEnv) 