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
    if method_name == 'PPO':
        from rllib.ppo import PPO as Method
        config.set('net_ac', rllib.ppo.ActorCriticContinuous)
    else:
        raise NotImplementedError('Not support this method.')

    model_name = Method.__name__ + '-' + config.env_name
    writer = rllib.basic.create_dir(config, model_name)
    method = Method(config, writer)
    return writer, env, method



def run_one_episode(i_episode, config, writer, env, method):
    running_reward = 0
    avg_length = 0
    state = env.reset()
    # while True:
    for i in range(config.epoch_length):
        action = method.select_action( torch.from_numpy(state).unsqueeze(0).float() )
        # next_state, reward, done, _ = env.step(action.cpu().numpy().squeeze())
        next_state, reward, done, _ = env.step(action.action.cpu().numpy().squeeze())

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
    
