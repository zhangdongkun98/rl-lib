
from abc import ABC, abstractmethod

import torch

from carla_utils.system import YamlConfig
from carla_utils.basic import Writer


class MethodSingleAgent(ABC):
    def __init__(self, config: YamlConfig, writer: Writer):
        self.method_name = config.method
        self.device = config.device
        self.path_pack = config.path_pack
        self.writer = writer

        self.dim_state, self.dim_action = config.dim_state, config.dim_action

        self.step_select = self.step_train = self.step_update = -1

        self.models_to_load, self.models_to_save = None, None

    def update_policy(self):
        self.step_update += 1
    def select_action(self):
        self.step_select += 1

    def _save_model(self):
        # print("[update_policy] save model")
        [model.save_model(self.path_pack.save_model_path, self.step_update) for model in self.models_to_save]
    def _load_model(self):
        print('[update_policy] load model')
        [model.load_model() for model in self.models_to_load]
        return

    def store(self, experience):
        self._replay_buffer.push(experience)

