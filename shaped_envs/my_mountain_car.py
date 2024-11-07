"""JAX compatible version of MountainCar-v0 OpenAI gym environment.


Source:
github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
"""

from typing import Any, Dict, Optional, Tuple, Union, TypeVar


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces

TEnvState = TypeVar("TEnvState", bound="EnvState")
TEnvParams = TypeVar("TEnvParams", bound="EnvParams")


@struct.dataclass
class EnvState(environment.EnvState):
    position: jnp.ndarray
    velocity: jnp.ndarray
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    min_position: float = -1.2
    max_position: float = 0.6
    max_speed: float = 0.07
    goal_position: float = 0.5
    goal_velocity: float = 0.0
    force: float = 0.001
    gravity: float = 0.0025
    max_steps_in_episode: int = 200


class MyMountainCar(environment.Environment[EnvState, EnvParams]):
    """JAX Compatible  version of MountainCar-v0 OpenAI gym environment."""

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step(
        self,
        key: chex.PRNGKey,
        state: TEnvState,
        action: Union[int, float, chex.Array],
        max_reached_pos: float,
        last_eval_goal_return: float,
        params: Optional[TEnvParams] = None,
    ) -> Tuple[chex.Array, TEnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(key, state, action, params, max_reached_pos, last_eval_goal_return)
        obs_re, state_re = self.reset_env(key_reset, params)
        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, done, info

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
        max_reached_pos: float,
        last_eval_goal_return: float,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Perform single timestep state transition."""
        velocity = (
            state.velocity
            + (action - 1) * params.force
            - jnp.cos(3 * state.position) * params.gravity
        )
        velocity = jnp.clip(velocity, -params.max_speed, params.max_speed)
        position = state.position + velocity
        position = jnp.clip(position, params.min_position, params.max_position)
        velocity = velocity * (1 - (position == params.min_position) * (velocity < 0))

        # Update state dict and evaluate termination conditions
        state = EnvState(position=position, velocity=velocity, time=state.time + 1)
        done = self.is_terminal(state, params)
        
        reward = -1.0

        phi_h_prime = (jnp.maximum(jnp.abs(-0.5-position), max_reached_pos))
        phi_h = max_reached_pos
        #bampf_reward = phi_h_prime
        #bampf_reward = 1.5*(0.99*phi_h_prime*(1 - done) - phi_h)
        bampf_reward = jnp.abs(-0.5-position)

        shaped_reward = reward + bampf_reward
        
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            jnp.array(shaped_reward),
            done,
            {"discount": self.discount(state, params), "shape_reward": bampf_reward, "real_reward": reward, "goal_reward": 0.0},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        init_state = jax.random.uniform(key, shape=(), minval=-0.6, maxval=-0.4)
        state = EnvState(position=init_state, velocity=jnp.array(0.0), time=0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Return observation from raw state trafo."""
        return jnp.array([state.position, state.velocity])

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        done1 = (state.position >= params.goal_position) * (
            state.velocity >= params.goal_velocity
        )

        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        done = jnp.logical_or(done1, done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "MountainCar-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 3

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(3)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        low = jnp.array(
            [params.min_position, -params.max_speed],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [params.max_position, params.max_speed],
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, (2,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        low = jnp.array(
            [params.min_position, -params.max_speed],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [params.max_position, params.max_speed],
            dtype=jnp.float32,
        )

        return spaces.Dict(
            {
                "position": spaces.Box(low[0], high[0], (), dtype=jnp.float32),
                "velocity": spaces.Box(low[1], high[1], (), dtype=jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )