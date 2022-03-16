
import numpy as np
import os
from os.path import join
import time
from collections import namedtuple
from tensorboardX import SummaryWriter

import torch

from .system import prefix
from .yaml import YamlConfig


class Writer(SummaryWriter):
    def __init__(self, **kwargs):
        super(Writer, self).__init__(**kwargs)

        description = kwargs['comment']
        self.add_text('description', description, 0)

        file_dir, file_name = os.path.split(self.file_writer.event_writer._ev_writer._file_name)
        dir_name = file_name.split('tfevents.')[-1].split('.')[0] + '--log'

        self.data_dir = join(file_dir, dir_name)
        os.makedirs(self.data_dir, exist_ok=True)

        self.data_cache = dict()
    
    
    def _clear_cache(self, tag):
        self.data_cache[tag]['count'] = 0
        self.data_cache[tag]['data'] = ''
    
    def _write_cache_to_disk(self, tag):
        file_path = join(self.data_dir, tag+'.txt')
        file_dir, _ = os.path.split(file_path)
        os.makedirs(file_dir, exist_ok=True)
        with open(file_path, mode='a') as f:
            f.write(self.data_cache[tag]['data'])
        
        self._clear_cache(tag)


    def add_scalar(self, *args):
        res = super().add_scalar(*args)

        tag, scalar_value, global_step = args[0], args[1], args[2]
        if isinstance(scalar_value, torch.Tensor):
            scalar_value = scalar_value.item()

        if tag not in self.data_cache:
            self.data_cache[tag] = dict()
            self._clear_cache(tag)
        self.data_cache[tag]['count'] += 1
        self.data_cache[tag]['data'] += str(global_step) + ' ' + str(scalar_value) + '\n'
        
        if self.data_cache[tag]['count'] % 100 == 0:
            self._write_cache_to_disk(tag)
        return res


    def close(self):
        res = super().close()

        for tag in self.data_cache:
            self._write_cache_to_disk(tag)
        return res


class PseudoWriter(object):
    def __init__(self, log_dir, comment, max_queue):
        self.log_dir = log_dir
        self.comment = comment
        self.max_queue = max_queue




PathPack = namedtuple('PathPack', ('log_path', 'save_model_path', 'output_path', 'code_path'))

def create_dir(config: YamlConfig, model_name, mode='train', writer_cls=Writer):
    '''
        create dir and save config
        Args:
            config: need to contain:
                config.description: str
    '''
    dataset_name = model_name + '/' + time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())) + '----' + str(config.description)
    if mode != 'train':
        dataset_name += '-' + mode
    print(prefix(__name__) + 'dir name: ', 'results/'+ dataset_name)
    work_path = os.getcwd()
    log_path = join(work_path, 'results', dataset_name, 'log')
    save_model_path = join(work_path, 'results', dataset_name, 'saved_models')
    output_path = join(work_path, 'results', dataset_name, 'output')
    code_path = join(work_path, 'results', dataset_name, 'code')
    os.makedirs(save_model_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(code_path, exist_ok=True)

    if config.get('github_repos', None) != None:
        codes = pack_code(config.github_repos)
        compress_code(code_path, codes)

    writer = writer_cls(log_dir=log_path, comment=dataset_name, max_queue=100)
    path_pack = PathPack(log_path, save_model_path, output_path, code_path)
    config.set('dataset_name', dataset_name)
    config.set('path_pack', path_pack)

    with open(join('results', dataset_name, 'comments'), mode='w', encoding='utf-8') as _: pass
    # config.save(join('results', dataset_name))
    return writer


def pack_code(repos):
    origin_dir = os.getcwd()
    import subprocess

    codes = {}
    for repo in repos:
        repo = os.path.expanduser(repo)
        os.chdir(repo)
        repo_name = repo.split('/')[-1]

        file1 = subprocess.getstatusoutput('git ls-files')[1].split('\n')
        filed = subprocess.getstatusoutput('git ls-files -d')[1].split('\n')
        file2 = subprocess.getstatusoutput('git ls-files --others --exclude-standard')[1].split('\n')
        if '' in file2: file2.remove('')
        files = list(set(file1 + file2) - set(filed))

        codes[repo_name] = {'path': repo, 'files': files}
        # print('[pack_code] packing repo: ', repo_name, repo)
        # for i in files: print(i)
        # print('\n\n\n')

    os.chdir(origin_dir)
    return codes


def compress_code(save_dir, codes):
    import zipfile
    for repo, files in codes.items():
        file_dir = files['path']
        with zipfile.ZipFile(join(save_dir, repo +'.zip'), mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            for file_name in files['files']:
                zf.write(join(file_dir, file_name), arcname=file_name)
    return



