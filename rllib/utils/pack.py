import rllib

import os


if __name__ == '__main__':
    save_path = os.path.expanduser('~')
    repo = os.getcwd()
    codes = rllib.basic.workspace.pack_code([repo])
    rllib.basic.workspace.compress_code(save_path, codes)

