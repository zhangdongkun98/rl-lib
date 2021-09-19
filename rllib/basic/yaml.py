
from typing import ValuesView
import yaml
from os.path import join


def parse_yaml_file(file_path):
    data = None
    with open(file_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def parse_yaml_file_unsafe(file_path):
    data = None
    with open(file_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    config = YamlConfig(**data)
    # config.set_path(os.path.abspath(file_path))
    return config


class YamlConfig(object):
    def __init__(self, **kwargs):
        for key in kwargs:
            if isinstance(kwargs[key], type(dict())):
                setattr(self, key, YamlConfig(**kwargs[key]))
            else:
                setattr(self, key, kwargs[key])
        return
    
    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except:
            e = 'yaml file has not attribute \'{}\''.format(name)
            raise AttributeError(e)
    
    def get(self, key, default):
        result = default
        if hasattr(self, key):
            result = getattr(self, key)
        return result
    
    def set(self, key, value):
        if hasattr(self, key):
            print('[YamlConfig] warning: cannot imagine why need this: set {} to {}'.format(key, str(value)))
            # raise NotImplementedError('warning: cannot imagine why need this.')
        setattr(self, key, value)
    
    def delete(self, key):
        if hasattr(self, key):
            delattr(self, key)

    
    def update(self, config):
        """
        
        Args:
            config: YamlConfig or argparse
        
        Returns:
            None
        """

        block_words = self._get_block_words()

        for attribute in dir(config):
            if attribute in block_words:
                if not isinstance(config, YamlConfig): print('[YamlConfig] ignore attribute: ', attribute)
                continue
            if not attribute.startswith('_'):
                setattr(self, attribute, getattr(config, attribute))
        return
    
    def _get_block_words(self):
        block_words = [attr for attr in dir(YamlConfig) if callable(getattr(YamlConfig, attr)) and not attr.startswith('_')]
        return block_words
    
    def to_dict(self):
        block_words = self._get_block_words()
        result = dict()
        for attribute in dir(self):
            if attribute in block_words: continue
            if not attribute.startswith('_'):
                value = getattr(self, attribute)
                if isinstance(value, YamlConfig):
                    result[attribute] = value.to_dict()
                else: result[attribute] = value
        return result

    def save(self, path):
        config_dict = self.to_dict()
        with open(join(path, 'config.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(data=config_dict, stream=f, allow_unicode=True)
        return
