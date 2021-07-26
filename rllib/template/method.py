
from abc import ABC, abstractproperty

import torch

from ..basic import YamlConfig
from ..basic import Writer

class MethodSingleAgent(ABC):
    def __init__(self, config: YamlConfig, writer: Writer):
        config.set('method_name', self.__class__.__name__)

        self.device = config.device
        self.path_pack = config.path_pack
        self.writer = writer

        self.dtype = torch.float32
        self.dim_state, self.dim_action = config.dim_state, config.dim_action
        self.step_select = self.step_train = self.step_update = -1

        self.models_to_load, self.models_to_save = None, None

        self._memory = None

    def update_parameters(self):
        self.step_update += 1
    def select_action(self):
        self.step_select += 1

    def _save_model(self):
        # print("[update_parameters] save model")
        [model.save_model(self.path_pack.save_model_path, self.step_update) for model in self.models_to_save]
    def _load_model(self):
        print('[update_parameters] load model')
        [model.load_model() for model in self.models_to_load]
        return

    def store(self, experience):
        self._memory.push(experience)

