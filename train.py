import os
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
import jax
import pathlib
from utils.models import get_model_ready
from utils.helpers import load_config, save_pkl_object

import gym

import wandb
import gymnax

import matplotlib.pyplot as plt
from matplotlib import animation


import gym
import numpy as np
from visualize import rollout_episode

np.bool8 = np.bool_





def main(config, mle_log, log_ext=""):
    """Run training with ES or PPO. Store logs and agent ckpt."""

    #os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".50"
    #os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    #os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]='platform'
    rng = jax.random.PRNGKey(config.seed_id)
    # Setup the model architecture
    rng, rng_init = jax.random.split(rng)
    model, params = get_model_ready(rng_init, config)
    wandb.init(project="BAMDPgym", config=config, name=config.exp_name)

    # Run the training loop (either evosax ES or PPO)
    if config.train_type == "ES":
        from utils.es import train_es as train_fn
    elif config.train_type == "PPO":
        from utils.ppo import train_ppo as train_fn
    else:
        raise ValueError("Unknown train_type. Has to be in ('ES', 'PPO').")

    # Log and store the results.
    log_steps, log_return, log_real_return, log_goal_return, log_shape_return, network_ckpt = train_fn(
        rng, config, model, params, mle_log
    )

    data_to_store = {
        "log_steps": log_steps,
        "log_return": log_return,
        "log_real_return": log_real_return,
        "log_goal_return": log_goal_return,
        "log_shape_return": log_shape_return,
        "network": network_ckpt,
        "train_config": config,
    }

    savedir = f"agents/{config.env_name}/training_logs/{config.train_type.lower()}{log_ext}"
    pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
    save_pkl_object(
        data_to_store,
        savedir+"/weights.pkl",
    )
    
    ### VISUALIZATION
    base_env_name = 'MountainCar-v0'#'MountainCarContinuous-v0'
    env, env_params = gymnax.make(base_env_name,  **config.train_config.env_kwargs,)
    env_params.replace(**config.train_config.env_params)

    state_seq, action_seq, cum_rewards = rollout_episode(env, env_params, model, network_ckpt)
    env = gym.make(base_env_name, render_mode="rgb_array")
    # Wrap the environment with the RecordVideo wrapper to record videos
    # The episode_trigger lambda function ensures that a video is recorded every 10 episodes
    env = gym.wrappers.RecordVideo(env, savedir, episode_trigger=lambda episode_id: episode_id == 0)
    _ = env.reset()
    action_seq = np.array(action_seq)
    for i in range(len(action_seq)):
        # Step the environment using the sampled action
        observation, reward, terminated, truncated, info = env.step(action_seq[i])
    
    # Properly close the environment to free resources
    env.close()

    wandb.log({"final_rollout": wandb.Video(savedir+"/rl-video-episode-0.mp4")})


if __name__ == "__main__":
    # Use MLE-Infrastructure if available (e.g. for parameter search)
    # try:
    #     from mle_toolbox import MLExperiment

    #     mle = MLExperiment(config_fname="configs/cartpole/ppo.yaml")
    #     main(mle.train_config, mle_log=mle.log)
    # # Otherwise use simple logging and config loading
    # except Exception:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config",
        "--config_fname",
        type=str,
        default="configs/CartPole-v1/ppo.yaml",
        help="Path to configuration yaml.",
    )
    parser.add_argument(
        "-seed",
        "--seed_id",
        type=int,
        default=0,
        help="Random seed of experiment.",
    )
    parser.add_argument(
        "-lr",
        "--lrate",
        type=float,
        default=5e-04,
        help="Learning rate of experiment.",
    )
    args, _ = parser.parse_known_args()
    config = load_config(args.config_fname, args.seed_id, args.lrate)
    main(
        config.train_config,
        mle_log=None,
        log_ext=config.train_config.exp_name,#str(args.lrate) if args.lrate != 5e-04 else "",
    )
