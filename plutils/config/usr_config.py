import os
from collections import OrderedDict

import ruamel.yaml
from ruamel.yaml.main import round_trip_dump as yaml_dump


def get_usr_config(path):
    ext = os.path.splitext(path)[1].lstrip('.')
    ext = ext.rstrip(' ')
    if ext == 'yaml':
        with open(path) as file:
            yml = ruamel.yaml.YAML()
            yml.allow_duplicate_keys = True
            doc = yml.load(file)
        usr_config = UsrConfigs(doc)
    else:
        raise NotImplementedError(f'Config file format .{ext} not understood')

    return usr_config


class EmptyConfig:
    def __init__(self):
        pass

    def __repr__(self):
        return 'empty'

    def __len__(self):
        return 0


class UsrConfigs:
    def __init__(self, obj={}):
        for k, v in obj.items():
            if isinstance(v, dict):
                setattr(self, k, UsrConfigs(v))
            elif isinstance(v, list):
                if len(v) == 0:
                    setattr(self, k, [])
                elif isinstance(v[0], OrderedDict):
                    setattr(self, k, [])
                    for m in v:
                        getattr(self, k).append(UsrConfigs(m))
                else:
                    setattr(self, k, [])
                    for m in v:
                        getattr(self, k).append(m)
            elif v is None:
                setattr(self, k, EmptyConfig())
            else:
                setattr(self, k, v)

    def __getitem__(self, val):
        return self.__dict__[val]

    def __len__(self):
        return len(self.__dict__)

    def save(self, dir_path):
        path = os.path.join(dir_path, 'usr_config.yaml')
        usr_config_dict = self.__get_dict__()

        with open(path, 'w') as file:
            yaml_dump(usr_config_dict, file, indent=4)

    def __get_dict__(self):
        ret = {}
        for k, v in self.__dict__.items():
            if isinstance(v, UsrConfigs):
                ret[k] = v.__get_dict__()
            elif isinstance(v, EmptyConfig):
                ret[k] = None
            elif isinstance(v, list):
                if len(v) == 0:
                    setattr(self, k, [])
                elif isinstance(v[0], UsrConfigs):
                    ret[k] = []
                    for item in v:
                        ret[k].append(item.__get_dict__())
                elif isinstance(v[0], float) or isinstance(v[0], int) or isinstance(v[0], str):
                    ret[k] = v
            else:
                ret[k] = v
        return ret

    def __repr__(self):
        return '{%s}' % str(', '.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))
