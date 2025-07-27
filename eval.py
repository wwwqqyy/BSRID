import argparse
import os
from collections import deque
from tqdm import tqdm

import colorama
import cv2
import gymnasium
import numpy as np
import torch
from einops import rearrange

import agents
import env_wrapper
from sub_models.world_models import WorldModel
from utils import seed_np_torch, load_config


def process_visualize(img):
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 640))
    return img


def build_single_env(env_name, image_size):
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    return env


def build_vec_env(env_name, image_size, num_envs):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size)

    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


def eval_episodes(num_episode, env_name, num_envs, context_length, image_size,
                  world_model: WorldModel, agent: agents.ActorCriticAgent, step):
    world_model.eval()
    agent.eval()
    vec_env = build_vec_env(env_name, image_size, num_envs=num_envs)
    print("Step: " + f"{step}" + " Current env: " + colorama.Fore.RED + f"{env_name}" + colorama.Style.RESET_ALL)
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()
    context_obs = deque(maxlen=context_length)
    context_action = deque(maxlen=context_length)

    final_rewards = []
    while True:
        # sample part >>>
        with torch.no_grad():
            if len(context_action) == 0:
                action = vec_env.action_space.sample()
            else:
                context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                model_context_action = np.stack(list(context_action), axis=1)
                model_context_action = torch.Tensor(model_context_action).cuda()
                prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent,
                                                                                         model_context_action)
                action = agent.sample_as_env_action(
                    torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                    greedy=False
                )

        context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W") / 255)
        context_action.append(action)

        obs, reward, done, truncated, info = vec_env.step(action)
        # cv2.imshow("current_obs", process_visualize(obs[0]))
        # cv2.waitKey(10)

        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            for i in range(num_envs):
                if done_flag[i]:
                    final_rewards.append(sum_reward[i])
                    sum_reward[i] = 0
                    if len(final_rewards) == num_episode:
                        print(
                            "Mean reward: " + colorama.Fore.RED + f"{np.mean(final_rewards)}" + colorama.Style.RESET_ALL)
                        return final_rewards

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        current_info = info
        # <<< sample part


if __name__ == "__main__":
    # ignore warnings
    import warnings

    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-run_name", type=str, required=True)
    args = parser.parse_args()
    conf = load_config(args.config_path)
    print(colorama.Fore.GREEN + str(args) + colorama.Style.RESET_ALL)
    # print(colorama.Fore.RED + str(conf) + colorama.Style.RESET_ALL)

    # set seed
    seed_np_torch(seed=conf.BasicSettings.Seed)

    # build and load model/agent
    import train

    dummy_env = build_single_env(args.env_name, conf.BasicSettings.ImageSize)
    action_dim = dummy_env.action_space.n
    world_model = train.build_world_model(conf, action_dim)
    agent = train.build_agent(conf, action_dim)

    # root_path = f"ckpt/{args.run_name}"
    root_path = f"weights/{args.run_name}"
    import glob

    paths = glob.glob(f"{root_path}/world_model_*.pth")
    steps = [int(path.split("_")[-1].split(".")[0]) for path in paths]
    steps.sort()
    
    # Number of times to repeat the evaluation process
    num_evaluation_runs = 3
    
    # Create a directory for evaluation results if it doesn't exist
    if not os.path.exists("eval_result_last"):
        os.mkdir("eval_result_last")
    
    # Create a progress bar for the evaluation runs (outer loop)
    print(f"Starting {num_evaluation_runs} evaluation runs...")
    run_progress_bar = tqdm(range(num_evaluation_runs), desc="Evaluation Runs", unit="run")
    
    # Store results for all runs
    all_results = {}
    
    for run_idx in run_progress_bar:
        # Update the progress bar description with current run index (1-based)
        run_progress_bar.set_postfix({"Current Run": f"{run_idx + 1}/{num_evaluation_runs}"})
        
        print(f"\nStarting evaluation run {run_idx + 1}/{num_evaluation_runs}")
        
        run_results = []
        for step in steps:
            world_model.load_state_dict(torch.load(f"{root_path}/world_model_{step}.pth"))
            agent.load_state_dict(torch.load(f"{root_path}/agent_{step}.pth"))
            
            # eval
            episode_return = eval_episodes(
                num_episode=20,
                env_name=args.env_name,
                num_envs=5,
                context_length=conf.JointTrain.ContextLength,
                image_size=conf.BasicSettings.ImageSize,
                world_model=world_model,
                agent=agent,
                step=step,
            )
            run_results.append([step, episode_return])
            
            # Save results for this run
            result_filename = f"eval_result_last/{args.run_name}_run{run_idx + 1}.txt"
            with open(result_filename, "w") as f:
                for step, episode_return in run_results:
                    f.write(f"{step},{episode_return}\n")
        
        print(f"Evaluation run {run_idx + 1}/{num_evaluation_runs} completed!")
    
    print("All evaluation runs completed!")
