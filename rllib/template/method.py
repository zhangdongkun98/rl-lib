
import torch
import os
from os.path import join
from typing import List
import time
import tqdm

from torch.optim import Optimizer

from ..basic import Data
from ..basic import YamlConfig
from ..basic import Writer
from ..basic import prefix
from .model import Model

class Method(object):
    def __init__(self, config: YamlConfig, writer: Writer, tag_name='method'):
        config.set('method_name', self.__class__.__name__)

        self.config = config

        self.device = config.device
        self.path_pack = config.path_pack
        self.writer = writer
        self.tag_name = tag_name
        self.log_dir = join(self.path_pack.log_path, tag_name)
        self.output_dir = join(self.path_pack.output_path, tag_name)
        self.model_dir = self.path_pack.save_model_path + '_' + tag_name
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.dtype = torch.float32
        self.step_train = self.step_update = -1

        self.models: List[Model] = []
        self.models_to_load, self.models_to_save = None, None
        self.optimizers: List[Optimizer] = []


    def _save_model(self, iter_num=None):
        if iter_num == None:
            iter_num = self.step_update
        [model.save_model(self.model_dir, iter_num) for model in self.models_to_save]
    def _load_model(self, model_num=None):
        [model.load_model(model_num=model_num) for model in self.models_to_load]
        return
    def load_model(self):
        return self._load_model()

    def update_parameters_start(self):
        self.step_update += 1

    def update_callback(self, local):
        local.pop('self')
        # local.pop('__class__')
        return Data(**local)

    def update_parameters_(self, index, n_iters=1000):
        t1 = time.time()

        # for i in range(n_iters):
        #     if i % (n_iters //10) == 0:
        #         print(prefix(self) + 'update_parameters index / total: ', i, n_iters)
        #     self.update_parameters()

        for i in tqdm.tqdm(range(n_iters)):
        # for i in range(n_iters):
            self.update_parameters()
        
        t2 = time.time()
        self.writer.add_scalar(f'{self.tag_name}/update_time', t2-t1, index)
        self.writer.add_scalar(f'{self.tag_name}/update_iters', n_iters, index)
        self.writer.add_scalar(f'{self.tag_name}/update_time_per_iter', (t2-t1) /n_iters, index)
        # print()
        return

    def update_parameters(self):
        return


    def get_writer(self):
        return self.writer
    def reset_writer(self):
        self.writer = Writer(log_dir=self.log_dir, comment=self.config.dataset_name, max_queue=100)

    def get_models(self):
        return self.models_to_save

    def get_model_params(self):
        models = self.get_models()
        model_params = [[param.data for param in model.parameters()] for model in models]
        return model_params

    def get_optimizers(self):
        return self.optimizers



class MethodSingleAgent(Method):
    def __init__(self, config: YamlConfig, writer: Writer, tag_name='method'):
        super(MethodSingleAgent, self).__init__(config, writer, tag_name)

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


    def get_buffer(self):
        return self.buffer

