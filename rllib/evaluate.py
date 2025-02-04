import rllib

from typing import Union

import torch


class EvaluateSingleAgent(rllib.template.MethodSingleAgent):
    def __init__(self, config, writer, tag_name='method'):
        super().__init__(config, writer, tag_name)

        self.config = config

        self.model_dir, self.model_num = config.model_dir, config.model_num

        method_name = config.method.upper()
        config.set('method_name', method_name)
        self.method_name = method_name

        self.select_method()
        self._load_model(self.model_num)
        return
    
    def select_method(self):
        config, method_name = self.config, self.method_name
        if method_name == 'PPO':
            from . import ppo_simple
            self.policy: Union[ppo_simple.ActorCriticDiscrete, ppo_simple.ActorCriticContinuous] = config.net_ac(config).to(self.device)
            self.models_to_load = [self.policy]
            self.select_action = self.select_action_ppo
        elif method_name == 'TD3':
            from . import td3
            self.critic = config.get('net_critic', td3.Critic)(config).to(self.device)
            self.actor = config.get('net_actor', td3.Actor)(config).to(self.device)
            self.models_to_load = [self.critic, self.actor]
            self.select_action = self.select_action_td3
        elif method_name == 'SAC':
            from . import sac
            self.critic = config.get('net_critic', sac.Critic)(config).to(self.device)
            self.actor = config.get('net_actor', sac.Actor)(config).to(self.device)
            self.models_to_load = [self.critic, self.actor]
            self.select_action = self.select_action_sac
        
        elif method_name == 'DIAYN':
            from . import sac
            from .exploration import diayn
            self.critic = config.get('net_critic', sac.Critic)(config).to(self.device)
            self.actor = config.get('net_actor', sac.Actor)(config).to(self.device)
            self.discriminator = config.get('net_actor', diayn.Discriminator)(config).to(self.device)
            self.models_to_load = [self.critic, self.actor, self.discriminator]
            self.update_discriminator = lambda x, y: None
            self.select_action = self.select_action_sac
        
        else:
            raise NotImplementedError('No such method: ' + str(method_name))
        return



    def store(self, experience):
        return


    @torch.no_grad()
    def select_action_ppo(self, state):
        self.select_action_start()
        
        state = state.to(self.device)
        action_data = self.policy(state)
        action_data.update(action=action_data['mean'])
        # print('action: ', action.cpu().data, 'mean: ', mean.cpu().data, 'value: ', state_value.item())
        return action_data.cpu()


    @torch.no_grad()
    def select_action_td3(self, state):
        self.select_action_start()

        state = state.to(self.device)
        action = self.actor(state)
        value = self.critic(state, action)

        # print('action: ', action.cpu(), 'value', value)
        # import pdb; pdb.set_trace()

        return action


    @torch.no_grad()
    def select_action_sac(self, state):
        self.select_action_start()
        state = state.to(self.device)
        action, logprob, mean = self.actor.sample(state)
        return action

