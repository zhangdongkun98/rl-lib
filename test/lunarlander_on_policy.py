import rllib

import gym
import torch

from rllib.args import generate_args


def init(config):
    ### common param
    seed = config.seed
    rllib.basic.setup_seed(config.seed)
    config.set('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    ### env param
    env_name = "LunarLander-v2"
    render = False
    solved_reward = 230
    max_episodes = 10000
    epoch_length = 300
    config.set('render', render)
    config.set('solved_reward', solved_reward)
    config.set('max_episodes', max_episodes)
    config.set('epoch_length', epoch_length)

    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    config.set('dim_state', env.observation_space.shape[0])
    config.set('dim_action', env.action_space.n)

    ### method param
    method_name = config.method.upper()
    if method_name == 'PPO':
        from rllib.ppo import PPO as Method
        config.set('net_ac', rllib.ppo.ActorCriticDiscrete)
    else:
        raise NotImplementedError('Not support this method.')
    model_name = Method.__name__ + '-' + env_name
    writer = rllib.basic.create_dir(config, model_name)
    method = Method(config, writer)
    return writer, env, method


def train(config, writer, env, method):
    for i_episode in range(config.max_episodes):
        running_reward = 0
        avg_length = 0
        state = env.reset()
        # while True:
        for i in range(config.epoch_length):
            action = method.select_action( torch.from_numpy(state).unsqueeze(0).float() )
            next_state, reward, done, _ = env.step(action.item())

            if i == config.epoch_length -1:
                done = True

            experience = rllib.template.Experience(
                    state=torch.from_numpy(state).float().unsqueeze(0),
                    action=action.cpu(), reward=reward, done=done)
            method.store(experience)

            state = next_state

            running_reward += reward
            avg_length += 1
            if config.render: env.render()
            if done: break
        
        method.update_parameters()

        ### indicate the task is solved
        if running_reward > config.solved_reward:
            print("########## Solved! ##########")
            
        ### logging
        print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
        writer.add_scalar('index/reward', running_reward, i_episode)
        writer.add_scalar('index/avg_length', avg_length, i_episode)


def main():
    config = rllib.basic.YamlConfig()
    args = generate_args()
    config.update(args)

    writer, env, method = init(config)
    try:
        train(config, writer, env, method)
    finally:
        writer.close()


if __name__ == '__main__':
    main()
    
