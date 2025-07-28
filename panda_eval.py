import argparse
import os
import pickle
from importlib import metadata

import torch


from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from panda_ball_env import PandaBallHitEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="panda_ball_v2")
    parser.add_argument("--ckpt", type=int, default=4999)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    env = PandaBallHitEnv(cfg=env_cfg,num_envs=1,  visible=True)
    
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)
            print(f"rews {rews}")


if __name__ == "__main__":
    main()
