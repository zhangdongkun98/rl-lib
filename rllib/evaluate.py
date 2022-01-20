import rllib

from typing import Union

import torch


class EvaluateSingleAgent(rllib.template.MethodSingleAgent):
    def __init__(self, config, writer):
        super(EvaluateSingleAgent, self).__init__(config, writer)

        self.config = config

        self.model_dir, self.model_num = config.model_dir, config.model_num

        method_name = config.method.upper()
        config.set('method_name', method_name)
        self.method_name = method_name

        self.select_method()
        self._load_model()
        return
    
    def select_method(self):
        config, method_name = self.config, self.method_name
        if method_name == 'PPO':
            from . import ppo
            self.policy: Union[ppo.ActorCriticDiscrete, ppo.ActorCriticContinuous] = config.net_ac(config).to(self.device)
            self.models_to_load = [self.policy]
            self.select_action = self.select_action_ppo
        elif method_name == 'TD3':
            from . import td3
            self.critic = config.get('net_critic', td3.Critic)(config).to(self.device)
            self.actor = config.get('net_actor', td3.Actor)(config).to(self.device)
            self.models_to_load = [self.critic, self.actor]
            self.select_action = self.select_action_td3
        else:
            raise NotImplementedError('No such method: ' + str(method_name))
        return



    def store(self, experience):
        return


    @torch.no_grad()
    def select_action_ppo(self, state):
        self.select_action_start()
        
        state = state.to(self.device)
        action, logprob, mean = self.policy(state)

        action_logprobs, state_value, dist_entropy = self.policy.evaluate(state, action)

        # print('action: ', action.cpu().data, 'mean: ', mean.cpu().data, 'value: ', state_value.item())
        # import pdb; pdb.set_trace()
        return mean.cpu().data


    @torch.no_grad()
    def select_action_td3(self, state):
        self.select_action_start()

        state = state.to(self.device)
        action = self.actor(state)
        value = self.critic(state, action)

        # print('action: ', action.cpu(), 'value', value)
        # import pdb; pdb.set_trace()

        return action
