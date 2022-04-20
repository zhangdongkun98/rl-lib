
from typing import ValuesView
import yaml
from os.path import join

from .system import get_class_name


def parse_yaml_file(file_path):
    with open(file_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def parse_yaml_file_unsafe(file_path):
    data = parse_yaml_file(file_path)
    config = YamlConfig(**data)
    return config


def recursive_print_dict(d, indent=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print("\t" * indent, f"{k}:")
            recursive_print_dict(v, indent+1)
        else:
            print("\t" * indent, f"{k}:{v}")
    return


def recursive_str_dict(d, indent=0):
    res = ""
    for k, v in d.items():
        if isinstance(v, dict):
            res += "\t" * indent + f"{k}: " + '\n'
            res += recursive_str_dict(v, indent+1)
        else:
            res += "\t" * indent + f"{k}: {v}" + '\n'
    return res



class YamlConfig(object):
    def __init__(self, **kwargs):
        for key in kwargs:
            if isinstance(kwargs[key], type(dict())):
                setattr(self, key, YamlConfig(**kwargs[key]))
            else:
                setattr(self, key, kwargs[key])
        return


    def __repr__(self):
        return recursive_str_dict(self.to_dict(), 1)


    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except:
            e = 'yaml file has not attribute \'{}\''.format(name)
            raise AttributeError(e)
    
    def get(self, key, default=None):
        result = default
        if hasattr(self, key):
            result = getattr(self, key)
        return result
    
    def set(self, key, value):
        # if hasattr(self, key):
        #     print('[{}.set] warning: cannot imagine why need this: set {} from {} to {}'.format(
        #         get_class_name(self), key, str(getattr(self, key)), str(value)
        #     ))
        #     # raise NotImplementedError('warning: cannot imagine why need this.')
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
                if not isinstance(config, YamlConfig): print('[{}.update] ignore attribute: '.format(get_class_name(self)), attribute)
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
