
import torch
import os
from os.path import join

from ..basic import Data
from ..basic import YamlConfig
from ..basic import Writer


class Method(object):
    def __init__(self, config: YamlConfig, writer: Writer):
        config.set('method_name', self.__class__.__name__)

        self.config = config

        self.device = config.device
        self.path_pack = config.path_pack
        self.writer = writer
        self.output_dir = join(self.path_pack.output_path, 'method')
        os.makedirs(self.output_dir)

        self.dtype = torch.float32
        self.step_train = self.step_update = -1

        self.models_to_load, self.models_to_save = None, None


    def _save_model(self):
        # print("[update_parameters] save model")
        [model.save_model(self.path_pack.save_model_path, self.step_update) for model in self.models_to_save]
    def _load_model(self):
        print('[update_parameters] load model')
        [model.load_model() for model in self.models_to_load]
        return

    def update_parameters(self):
        self.step_update += 1

    def update_callback(self, local):
        local.pop('self')
        local.pop('__class__')
        return Data(**local)


class MethodSingleAgent(Method):
    def __init__(self, config: YamlConfig, writer: Writer):
        super(MethodSingleAgent, self).__init__(config, writer)

        self.dim_state, self.dim_action = config.dim_state, config.dim_action
        self.step_select = -1

        self._memory = None

    def select_action(self):
        self.step_select += 1

    def store(self, experience):
        self._memory.push(experience)


