
import os, glob
import numpy as np

import visdom

from ..basic import Writer
from ..basic import YamlConfig

def generate_args():
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('--dir', default='./', type=str, help='')

    args = argparser.parse_args()
    return args



def ls_files(specified_dir):
    files = []
    if os.path.isfile(specified_dir):
        files.append(specified_dir)
    else:
        for children in os.listdir(specified_dir):
            files.extend( ls_files(os.path.join(specified_dir, children)) )
    return files




def load_scalar(file_path):  ### ! todo delete
    print('[load_scalar] ', file_path)
    # data = np.loadtxt(file_path, delimiter=" ", converters={1: lambda x: x[1:-1]})
    try:
        data = np.loadtxt(file_path, delimiter=' ')
    except:
        data = np.loadtxt(file_path, delimiter=" tensor", converters={1: lambda x: x[1:-1]})

    return data


if __name__ == "__main__":
    config = YamlConfig()
    args = generate_args()
    config.update(args)

    files = ls_files(os.path.expanduser(config.dir))

    interested_dirs = []
    for file in files:
        file_dir, file_name = os.path.split(file)
        if 'events.out.tfevents' in file_name:  ### rewrite if needed
            interested_dirs.append(file_dir)


    import rllib; rllib.basic.setup_seed(2)  ### ! todo delete


    # from visdom import server
    # server.start_server()
    interested_dirs = interested_dirs[:3]


    for interested_dir in interested_dirs:
        print(interested_dir)

        vis = visdom.Visdom(env=interested_dir.replace('/', '_').replace('.', '_'))


        assert len(glob.glob(os.path.join(interested_dir, '*/'))) == 1
        log_dir = glob.glob(os.path.join(interested_dir, '*/'))[0]

        import random
        indices_dir = random.choice(glob.glob(os.path.join(interested_dir, '*/*')))
        indices = os.listdir(indices_dir)
        index = random.choice(indices)

        data = load_scalar(os.path.join(indices_dir, index))

        vis.line(X=data[:,0], Y=data[:,1])
        # import pdb; pdb.set_trace()


        # os.path.splitext(a)[0]


        print(os.listdir(random.choice(glob.glob(os.path.join(interested_dir, '*/*')))))

        print()



    # for i in files:
    #     print(i)
    print('len: ', len(interested_dirs))

