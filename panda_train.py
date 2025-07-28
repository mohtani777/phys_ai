import argparse
import os
import pickle
import shutil
from importlib import metadata


from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from panda_pick_and_place import PegInsertEnv

#from gpt_panda_ball_env import PandaBallHitEnv
#from gmn_panda_ball_env import PandaBallHitEnv
from panda_ball_env import PandaBallHitEnv

#from panda_ball_env_vf1 import PandaBallHitEnv

def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.3,  #org 0.2
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.995, #org 0.99
            "lam": 0.98, #org 0.95
            "learning_rate": 0.001, #org 0.001
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4, #org 4
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict

# 1.571, -1.7428, 2.0, -1.8292, -1.5705, 0.6548, 0.02, 0.02
def get_cfgs():
    env_cfg = {
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 2.0,
        "resampling_time_s": 4.0,
        "action_scale": 1.0,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
    }


    return env_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="panda_ball_v2")
    parser.add_argument("-B", "--num_envs", type=int, default=1048) #frg 512
    parser.add_argument("--max_iterations", type=int, default=5000)
    args = parser.parse_args()

    gs.init(backend=gs.gpu,logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg= get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump([env_cfg, train_cfg], open(f"{log_dir}/cfgs.pkl", "wb"), )

    env = PandaBallHitEnv(num_envs=args.num_envs,cfg=env_cfg,visible=False)
    
    env.reset()
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
