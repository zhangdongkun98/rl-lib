import rllib

import gym
import torch
import torch.nn as nn



def init_ppo(config: rllib.basic.YamlConfig, seed):
    env_name = "LunarLander-v2"
    env = gym.make(env_name)
    env.seed(seed)

    setattr(env, 'dim_state', env.observation_space.shape[0])
    setattr(env, 'dim_action', 4)

    import rllib
    from rllib.ppo import PPO

    config.set('dim_state', env.dim_state)
    config.set('dim_action', env.dim_action)
    config.set('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    config.set('net_ac', rllib.ppo.ActorCriticDiscrete)

    model_name = PPO.__name__ + '-' + env_name
    writer = rllib.basic.create_dir(config, model_name)
    method = PPO(config, writer)

    return writer, env, method


