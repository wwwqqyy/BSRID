import argparse
import os
import shutil
from collections import deque

import colorama
import gymnasium
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

import agents
import env_wrapper
from eval_script import run_eval
from replay_buffer import ReplayBuffer
from sub_models.world_models import WorldModel
from utils import seed_np_torch, Logger, load_config
import math


def newton_cooling_analytic(T0, Tenv, k, t):
    """
    牛顿冷却定律
    :param T0: 初始温度
    :param Tenv: 环境温度
    :param k: 冷却系数
    :param t: 时间
    :return: 温度
    """
    return Tenv + (T0 - Tenv) * math.exp(-k * t)


def build_single_env(env_name, image_size, seed):
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = env_wrapper.SeedEnvWrapper(env, seed=seed)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = env_wrapper.LifeLossInfo(env)
    return env


def build_vec_env(env_name, image_size, num_envs, seed):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size, seed)

    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


def train_world_model_step(replay_buffer: ReplayBuffer, world_model: WorldModel, batch_size,
                           batch_length, balanced_sample, temperature, logger):
    if balanced_sample:
        obs, action, reward, termination = replay_buffer.sample_balanced(batch_size, batch_length, temperature)
    else:
        obs, action, reward, termination = replay_buffer.sample(batch_size, batch_length)
    world_model.update(obs, action, reward, termination, logger=logger)


@torch.no_grad()
def world_model_imagine_data(replay_buffer: ReplayBuffer,
                             world_model: WorldModel, agent: agents.ActorCriticAgent,
                             imagine_batch_size,
                             imagine_context_length, imagine_batch_length,
                             log_video, balanced_sample, temperature, logger):
    """
    Sample context from replay buffer, then imagine data with world model and agent
    """
    world_model.eval()
    agent.eval()
    if balanced_sample:
        sample_obs, sample_action, sample_reward, sample_termination = replay_buffer.sample_balanced(
            imagine_batch_size, imagine_context_length, temperature)
    else:
        sample_obs, sample_action, sample_reward, sample_termination = replay_buffer.sample(
            imagine_batch_size, imagine_context_length)

    latent, action, reward_hat, termination_hat = world_model.imagine_data(
        agent, sample_obs, sample_action,
        imagine_batch_size=imagine_batch_size,
        imagine_batch_length=imagine_batch_length,
        log_video=log_video,
        logger=logger
    )
    return latent, action, None, None, reward_hat, termination_hat


def joint_train_world_model_agent(env_name, context_length, balanced_sample, rid, temperature, max_steps, num_envs,
                                  image_size,
                                  replay_buffer: ReplayBuffer,
                                  world_model: WorldModel,
                                  agent: agents.ActorCriticAgent,
                                  batch_size, batch_length,
                                  imagine_batch_size,
                                  imagine_context_length, imagine_batch_length,
                                  save_every_steps, seed, logger):
    # create ckpt dir
    os.makedirs(f"ckpt/{args.n}", exist_ok=True)

    # build vec env, not useful in the Atari 100k setting
    # but when the max_steps is large, you can use parallel envs to speed up
    vec_env = build_vec_env(env_name, image_size, num_envs=num_envs, seed=seed)
    print("Current env: " + colorama.Fore.GREEN + f"{env_name}" + colorama.Style.RESET_ALL)

    # reset envs and variables
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()
    context_obs = deque(maxlen=context_length)
    context_action = deque(maxlen=context_length)

    Tenv = temperature
    # sample and train
    for step in tqdm(range(max_steps // num_envs)):
        # sample part >>>
        if replay_buffer.ready():

            world_model.eval()
            agent.eval()
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
        else:

            action = vec_env.action_space.sample()

        obs, reward, done, truncated, info = vec_env.step(action)

        replay_buffer.append(current_obs, action, reward, np.logical_or(done, info["life_loss"]))

        done_flag = np.logical_or(done, truncated)

        if done_flag.any():
            for i in range(num_envs):  # num_envs=1
                if done_flag[i]:  # True game over
                    logger.log(f"sample/{env_name}_reward", sum_reward[i])
                    sum_reward[i] = 0

        # update current_obs, current_info and sum_reward
        sum_reward += reward

        current_obs = obs
        current_info = info
        # <<< sample part

        # train world model part >>>
        if replay_buffer.ready():
            train_world_model_step(
                replay_buffer=replay_buffer,
                world_model=world_model,
                batch_size=batch_size,
                batch_length=batch_length,
                balanced_sample=balanced_sample,
                temperature=temperature,
                logger=logger
            )
        # <<< train world model part

        # train agent part >>>
        if replay_buffer.ready():
            # save_every_steps=2500
            if step % (save_every_steps // num_envs) == 0:
                log_video = True
            else:
                log_video = False

            # return latent, action, None, None, reward_hat, termination_hat
            imagine_latent, agent_action, _, _, imagine_reward, imagine_termination = \
                world_model_imagine_data(
                    replay_buffer=replay_buffer,
                    world_model=world_model,
                    agent=agent,
                    imagine_batch_size=imagine_batch_size,
                    imagine_context_length=imagine_context_length,
                    imagine_batch_length=imagine_batch_length,
                    log_video=log_video,
                    balanced_sample=balanced_sample,
                    temperature=temperature,
                    logger=logger
                )

            for i in range(rid):
                agent.update(
                    latent=imagine_latent,
                    action=agent_action,
                    reward=imagine_reward,
                    termination=imagine_termination,
                    logger=logger
                )
        # <<< train agent part

        # save model and eval per episode
        if step % (save_every_steps // num_envs) == 0:
            temperature = math.ceil(newton_cooling_analytic(200000, Tenv, 0.1, (step // save_every_steps + 1)))
            print(f"\nTemperature: {temperature}" + f"\nThe number of times of reusing imagined data: {rid}")
            print(colorama.Fore.GREEN + f"\nSaving model at total steps {step}" + colorama.Style.RESET_ALL)
            torch.save(world_model.state_dict(), f"ckpt/{args.n}/world_model_{step}.pth")
            torch.save(agent.state_dict(), f"ckpt/{args.n}/agent_{step}.pth")


def build_world_model(conf, action_dim):
    return WorldModel(
        in_channels=conf.Models.WorldModel.InChannels,
        action_dim=action_dim,
        transformer_hidden_dim=conf.Models.WorldModel.TransformerHiddenDim,
        transformer_num_heads=conf.Models.WorldModel.TransformerNumHeads,
        transformer_max_length=conf.Models.WorldModel.TransformerMaxLength,
        transformer_num_layers=conf.Models.WorldModel.TransformerNumLayers,
    ).cuda()


def build_agent(conf, action_dim):
    return agents.ActorCriticAgent(
        feat_dim=32 * 32 + conf.Models.WorldModel.TransformerHiddenDim,
        num_layers=conf.Models.Agent.NumLayers,
        hidden_dim=conf.Models.Agent.HiddenDim,
        action_dim=action_dim,
        gamma=conf.Models.Agent.Gamma,
        lambd=conf.Models.Agent.Lambda,
        entropy_coef=conf.Models.Agent.EntropyCoef,
    ).cuda()


if __name__ == "__main__":
    # ignore warnings
    import warnings

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, required=True)
    parser.add_argument("-seed", type=int, required=True)
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    args = parser.parse_args()
    conf = load_config(args.config_path)

    print(colorama.Fore.GREEN + str(args) + colorama.Style.RESET_ALL)
    print(colorama.Fore.RED + str(conf) + colorama.Style.RESET_ALL)
    # set seed
    seed_np_torch(seed=args.seed)

    # tensorboard writer
    logger = Logger(path=f"runs/{args.n}")
    # copy config file
    shutil.copy(args.config_path, f"runs/{args.n}/config.yaml")
    if conf.Task == "BSRID":
        # getting action_dim with dummy env
        dummy_env = build_single_env(args.env_name, conf.BasicSettings.ImageSize, seed=0)
        action_dim = dummy_env.action_space.n

        # build world model and agent
        world_model = build_world_model(conf, action_dim)

        agent = build_agent(conf, action_dim)

        # build replay buffer
        replay_buffer = ReplayBuffer(
            obs_shape=(conf.BasicSettings.ImageSize, conf.BasicSettings.ImageSize, 3),  # 64, 64, 3
            num_envs=conf.JointTrain.NumEnvs,
            max_length=conf.JointTrain.BufferMaxLength,  # 100000
            warmup_length=conf.JointTrain.BufferWarmUp,  # 1024
            store_on_gpu=conf.BasicSettings.ReplayBufferOnGPU  # True
        )

        # train
        joint_train_world_model_agent(
            env_name=args.env_name,
            context_length=conf.JointTrain.ContextLength,
            balanced_sample=conf.JointTrain.BalancedSample,
            rid=conf.JointTrain.RID,
            temperature=conf.JointTrain.Temperature,
            max_steps=conf.JointTrain.SampleMaxSteps,
            num_envs=conf.JointTrain.NumEnvs,
            image_size=conf.BasicSettings.ImageSize,
            replay_buffer=replay_buffer,
            world_model=world_model,
            agent=agent,
            batch_size=conf.JointTrain.BatchSize,
            batch_length=conf.JointTrain.BatchLength,
            imagine_batch_size=conf.JointTrain.ImagineBatchSize,
            imagine_context_length=conf.JointTrain.ImagineContextLength,
            imagine_batch_length=conf.JointTrain.ImagineBatchLength,
            save_every_steps=conf.JointTrain.SaveEverySteps,
            seed=args.seed,
            logger=logger
        )
    else:
        raise NotImplementedError(f"Task {conf.Task} not implemented")

    # game_name = args.n
    # game_name = game_name.split('-')
    # run_eval(game_name[0])
