

def generate_args():
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('-d', dest='description', default='Nothing', help='[Method] description.')


    argparser.add_argument('-e', '--env', default='LunarLander-v2', type=str, help='[Env] Available envs, including \
        LunarLander-v2 \
        LunarLanderContinuous-v2 \
        HalfCheetah-v2 \
        Hopper-v2 \
    .')
    argparser.add_argument('-m', '--method', default='None', type=str, help='[Method] Method to use.')

    argparser.add_argument('--model-dir', default='None', type=str, help='[Model] dir contains model (default: False)')
    argparser.add_argument('--model-num', default=-1, type=str, help='[Model] model-num to use.')

    argparser.add_argument('--seed', default=0, type=int, help='seed.')
    argparser.add_argument('--render', action='store_true', help='render the env (default: False)')

    ### method params
    argparser.add_argument('--batch-size', default=32, type=int, help='[Method Param]')
    argparser.add_argument('--buffer-size', default=2000, type=int, help='[Method Param]')

    argparser.add_argument('--weight-value', default=0.005, type=float, help='[Method Param] available: PPO')
    argparser.add_argument('--weight-entropy', default=0.001, type=float, help='[Method Param] available: PPO')

    args = argparser.parse_args()
    return args





class EnvParams(object):
    def __init__(self, env):
        # self.env_name = str(env).split('<')[-1][:-3]
        self.env_name = get_env_name(env)
        if self.env_name == 'LunarLander-v2':
            self.dim_state = env.observation_space.shape[0]
            self.dim_action = env.action_space.n
            self.solved_reward = 230
            self.time_tolerance = 300
            self.continuous_action_space = False
        elif self.env_name == 'LunarLanderContinuous-v2':
            self.dim_state = env.observation_space.shape[0]
            self.dim_action = env.action_space.shape[0]
            self.solved_reward = 230
            self.time_tolerance = 300
            self.continuous_action_space = True

        elif self.env_name == 'HalfCheetah-v2':
            self.dim_state = env.observation_space.shape[0]
            self.dim_action = env.action_space.shape[0]
            self.solved_reward = 500
            self.time_tolerance = 1000 # int(3e10)
            self.continuous_action_space = True
        elif self.env_name == 'Hopper-v2':
            self.dim_state = env.observation_space.shape[0]
            self.dim_action = env.action_space.shape[0]
            self.solved_reward = 1000
            self.time_tolerance = 1000 # int(3e10)
            self.continuous_action_space = True

        else:
            raise RuntimeError(f'Not support this env: {self.env_name}')
        return



def get_env_name(env):
    if hasattr(env, 'env'):
        return get_env_name(env.env)
    
    if env.spec is None:
        return type(env).__name__
    else:
        return env.spec.id

