import rllib

import torch


class EvaluateSingleAgent(rllib.template.MethodSingleAgent):  ## TODO
    def __init__(self, config, path_pack, name):
        super(EvaluateSingleAgent, self).__init__(config, path_pack)

        method_name = name

        if method_name == 'PPO':
            from . import PPO
            self.ac = config.net_ac(config).to(self.device)
            self.actor = self.ac.actor
            self.models_to_load = [self.ac]
        else:
            raise NotImplementedError('No such method.')
        
        self._load_model()
        return
    

    @torch.no_grad()
    def select_action(self, state):
        super().select_action()
        action = self.actor(state.to(self.device))
        return action.cpu().data

