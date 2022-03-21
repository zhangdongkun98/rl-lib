
import torch
import os
from os.path import join
from typing import List

from ..basic import Data
from ..basic import YamlConfig
from ..basic import Writer
from ..basic import prefix
from .model import Model

class Method(object):
    def __init__(self, config: YamlConfig, writer: Writer):
        config.set('method_name', self.__class__.__name__)

        self.config = config

        self.device = config.device
        self.path_pack = config.path_pack
        self.writer = writer
        self.output_dir = join(self.path_pack.output_path, 'method')
        os.makedirs(self.output_dir, exist_ok=True)

        self.dtype = torch.float32
        self.step_train = self.step_update = -1

        self.models: List[Model] = []
        self.models_to_load, self.models_to_save = None, None


    def _save_model(self, iter_num=None):
        if iter_num == None:
            iter_num = self.step_update
        [model.save_model(self.path_pack.save_model_path, iter_num) for model in self.models_to_save]
    def _load_model(self):
        [model.load_model() for model in self.models_to_load]
        return

    def update_parameters_start(self):
        self.step_update += 1

    def update_callback(self, local):
        local.pop('self')
        # local.pop('__class__')
        return Data(**local)

    def update_parameters_(self, n_iters=1000):
        print(prefix(self) + 'total iters: ', n_iters)
        for i in range(n_iters):
            if i % (n_iters //10) == 0:
                print(prefix(self) + 'update_parameters i: ', i)
            self.update_parameters()
        print()
        return

    def update_parameters(self):
        return


    def get_writer(self):
        return self.writer
    def reset_writer(self):
        self.writer = Writer(log_dir=self.config.path_pack.log_path, comment=self.config.dataset_name, max_queue=100)


class MethodSingleAgent(Method):
    def __init__(self, config: YamlConfig, writer: Writer):
        super(MethodSingleAgent, self).__init__(config, writer)

        self.dim_state, self.dim_action = config.dim_state, config.dim_action
        self.step_select = -1

        self.buffer = None

    def select_action_start(self):
        self.step_select += 1

    def store(self, experience, **kwargs):
        self.buffer.push(experience, **kwargs)

    def close(self):
        self.writer.close()
        if hasattr(self.buffer, 'close'):
            self.buffer.close()
        return

