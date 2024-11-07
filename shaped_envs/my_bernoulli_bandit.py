"""JAX version of a Bernoulli bandit environment as in Wang et al. 2017."""

from typing import Any, Dict, Optional, Tuple, Union, TypeVar


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces
import functools
import numpy as np

TEnvState = TypeVar("TEnvState", bound="EnvState")
TEnvParams = TypeVar("TEnvParams", bound="EnvParams")

@struct.dataclass
class EnvState(environment.EnvState):
    last_action: jnp.ndarray
    last_reward: jnp.ndarray
    exp_reward_best: jnp.ndarray
    reward_probs: chex.Array
    time: float


@struct.dataclass
class EnvParams(environment.EnvParams):
    reward_prob: float = 0.1
    normalize_time: bool = True
    max_steps_in_episode: float = 100.0
    min_lim: float = -1.0
    max_lim: float = 1.0
    t_max: int = 100


class MyBernoulliBandit(environment.Environment[EnvState, EnvParams]):
    """JAX version of a Bernoulli bandit environment as in Wang et al. 2017."""

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: TEnvState,
        action: Union[int, float, chex.Array],
        arms_tried: chex.Array,
        params: Optional[TEnvParams] = None,
    ) -> Tuple[chex.Array, TEnvState, jnp.ndarray, jnp.ndarray, chex.Array, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, arms_tried, info = self.step_env(key, state, action, params, arms_tried)
        obs_re, state_re = self.reset_env(key_reset, params)
        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, done, arms_tried, info

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
        arms_tried: chex.Array,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, chex.Array, Dict[Any, Any]]:
        """Sample bernoulli reward, increase counter, construct input."""
        #phi_h_2 = jax.lax.select(state.last_reward, 1 - jnp.abs(action - state.last_action), jnp.abs(action - state.last_action))
        reward = jax.random.bernoulli(key, state.reward_probs[action]).astype(jnp.int32)
        state = EnvState(
            last_action=action,
            last_reward=reward,
            exp_reward_best=state.exp_reward_best,
            reward_probs=state.reward_probs,
            time=state.time + 1,
        )

        done = self.is_terminal(state, params)

        phi_h = jnp.sum(jnp.array(arms_tried))
        
        new_arms_tried = jnp.array(arms_tried) + jax.lax.select(action, jnp.array([0,1]),jnp.array([1,0]))
        new_arms_tried = jnp.clip(new_arms_tried,0,1)
        new_phi_h = jnp.sum(new_arms_tried)
        bampf = 0#(1 - done)*0.8*new_phi_h - phi_h
        non_bampf = new_phi_h - phi_h
        shaped_reward = reward + non_bampf

        
        return (
            lax.stop_gradient(self.get_obs(state, params)),
            lax.stop_gradient(state),
            shaped_reward,
            done,
            new_arms_tried,
            {"discount": self.discount(state, params), "real_reward": reward, "bampf_reward":bampf},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        # Sample reward function + construct state as concat with timestamp
        p1 = jax.random.choice(
            key,
            jnp.array([params.reward_prob, 1 - params.reward_prob]),
            shape=(1,),
        ).squeeze()

        state = EnvState(
            last_action=jnp.array(0),
            last_reward=jnp.array(0),
            exp_reward_best=jax.lax.select(p1 > 0.5, p1, 1 - p1),
            reward_probs=jnp.array([p1, 1 - p1]),
            time=0.0,
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> chex.Array:
        """Concatenate reward, one-hot action and time stamp."""
        action_one_hot = jax.nn.one_hot(state.last_action, 2).squeeze()
        time_rep = jax.lax.select(
            params.normalize_time,
            time_normalization(
                state.time, params.min_lim, params.max_lim, params.t_max
            ),
            state.time,
        )
        return jnp.hstack([state.last_reward, action_one_hot, time_rep])

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
        return jnp.array(done)

    @property
    def name(self) -> str:
        """Environment name."""
        return "BernoulliBandit-misc"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        low = jnp.array(
            [
                0,
                0,
                0,
                jax.lax.select(params.normalize_time, params.min_lim, 0.0),
            ],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [
                self.num_actions,
                1,
                1,
                jax.lax.select(
                    params.normalize_time,
                    params.max_lim,
                    params.max_steps_in_episode,
                ),
            ],
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, (4,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "last_action": spaces.Discrete(self.num_actions),
                "last_reward": spaces.Discrete(2),
                "exp_reward_best": spaces.Box(0, 1, (2,), jnp.float32),
                "reward_probs": spaces.Box(0, 1, (2,), jnp.float32),
                "time": spaces.Discrete(int(params.max_steps_in_episode)),
            }
        )


def time_normalization(
    t: float,
    min_lim: float = -1.0,
    max_lim: float = 1.0,
    t_max: int = 100,
) -> float:
    """Normalize time integer into range given max time."""
    return (max_lim - min_lim) * t / t_max + min_lim
