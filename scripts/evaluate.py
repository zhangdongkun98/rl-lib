import rllib

import gym
import torch

from utils.args import generate_args, EnvParams


def init(config):
    ### common param
    seed = config.seed
    rllib.basic.setup_seed(config.seed)
    config.set('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    ### env param
    env = gym.make(config.env)
    env.seed(seed)
    env.action_space.seed(seed)

    config.update(EnvParams(env))

    ### method param
    method_name = config.method.upper()
    if method_name == 'SAC':
        from rllib.sac import SAC as Method
    elif method_name == 'DDPG':
        from rllib.ddpg import DDPG as Method
    elif method_name == 'TD3':
        from rllib.td3 import TD3 as Method
    
    elif method_name == 'DQN':
        from rllib.dqn import DQN as Method

    elif method_name == 'DIAYN':
        config.set('num_skills', 6)
        config.dim_state += 1
        from rllib.exploration.diayn import DIAYN as Method

    else:
        raise NotImplementedError('Not support this method: {}.'.format(method_name))
    model_name = Method.__name__ + '-' + config.env_name
    Method = rllib.evaluate.EvaluateSingleAgent

    writer = rllib.basic.create_dir(config, model_name, mode='evaluate')
    method = Method(config, writer)
    if method_name == 'DIAYN':
        from rllib.exploration.diayn import EnvWrapper
        env = EnvWrapper(env, method, config.num_skills, mode='evaluate')
    return writer, env, method


def run_one_episode(i_episode, config, writer, env, method):
    running_reward = 0
    avg_length = 0
    state = env.reset()
    if hasattr(env, 'skill'):
        print('\nskill: ', env.skill)
    while True:
        action = method.select_action( torch.from_numpy(state).unsqueeze(0).float() )
        next_state, reward, done, _ = env.step( action.cpu().numpy().squeeze() )

        state = next_state

        running_reward += reward
        avg_length += 1
        if config.render: env.render()
        if done: break
    
    ### indicate the task is solved
    if running_reward > config.solved_reward:
        print("########## Solved! ##########")
    
    ### logging
    print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
    writer.add_scalar('env/reward', running_reward, i_episode)
    writer.add_scalar('env/avg_length', avg_length, i_episode)
    return


def main():
    config = rllib.basic.YamlConfig()
    args = generate_args()
    config.update(args)

    writer, env, method = init(config)
    try:
        for i_episode in range(10000):
            run_one_episode(i_episode, config, writer, env, method)
    finally:
        writer.close()


if __name__ == '__main__':
    main()
    
