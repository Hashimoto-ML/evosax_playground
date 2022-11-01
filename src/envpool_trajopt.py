import argparse 
import random 
from distutils.util import strtobool

import envpool
import numpy as np
import jax

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="evosax-trajopt",
        help="the wandb's project name")
    parser.add_argument("--num_envs", type=int, default=8,
        help="the number of parallel game environments")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.track:
        import wandb 

        wandb.init(
            project=args.wandb_project_name,
            sync_tensorboard=True,
            config=vars(args),
            monitor_gym=True,
            save_code=True
        )
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space 
    envs.is_vector_env = True 
    handle, recv, send, step_env = envs.xla()

    