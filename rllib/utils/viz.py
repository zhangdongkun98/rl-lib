
import os, glob
import numpy as np
import time
import re
import matplotlib.pyplot as plt
import seaborn as sns

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




def smooth(scalars, alpha):
    mean, variance = [], []
    m_old, v_old = scalars[0], 0.0
    for s in scalars:
        diff = s - m_old
        incr = alpha * diff
        m = m_old + incr
        v = (1- alpha) * (v_old + diff * incr)
        mean.append(m)
        variance.append(v)
        m_old, v_old = m, v
    return np.array(mean), np.array(variance)


colors = ['cornflowerblue', 'limegreen', 'darkorange']




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


    '''find all interested dirs'''
    for i, interested_dir in enumerate(interested_dirs):
        print(i, interested_dir)

    if len(interested_dirs) > 1:
        interested_dir_index = int(input('choose interested dir: '))
    else:
        interested_dir_index = 0
    interested_dir = interested_dirs[interested_dir_index]
    print('\ninterested dir is: ', interested_dir, '\n')


    '''look interested dir'''
    assert len(glob.glob(os.path.join(interested_dir, '*/'))) == 1
    log_dir = glob.glob(os.path.join(interested_dir, '*/'))[0]


    data_paths = ls_files(log_dir)
    data_paths = sorted(ls_files(log_dir), key=lambda p: int(re.findall('\d+', os.path.split(p)[-1])[0]) if len(re.findall('\d+', os.path.split(p)[-1])) > 0 else -1)
    data_paths = sorted(data_paths, key=lambda p: os.path.split(p)[-1].split(re.findall('\d+', os.path.split(p)[-1])[0] if len(re.findall('\d+', os.path.split(p)[-1])) > 0 else '-1')[0])
    for i, data_path in enumerate(data_paths):
        print(i, data_path)


    while True:
        interested_data_path_indices = eval(input('choose data paths: ').replace(' ', ','))
        if isinstance(interested_data_path_indices, int):
            interested_data_path_indices = (interested_data_path_indices,)
        assert isinstance(interested_data_path_indices, tuple)


        sns.set_style("darkgrid")
        fig = plt.figure(figsize=(15,8+2*len(interested_data_path_indices)), dpi=100)
        fig.suptitle(log_dir)
        axes = fig.subplots(len(interested_data_path_indices), 1)
        if len(interested_data_path_indices) == 1:
            axes = [axes,]

        for ax, interested_data_path_index in zip(axes, interested_data_path_indices):
            interested_data_path = data_paths[interested_data_path_index]

            t1 = time.time()
            data = np.loadtxt(interested_data_path, delimiter=' ').T
            data_step, data_value = data[0], data[1]
            t2 = time.time()
            data_mean, _ = smooth(data_value, 0.001)
            print('load time: ', t2-t1)

            ax.set_ylabel(os.path.split(interested_data_path)[-1][:-4])
            # ax.plot(data_step, data_mean, color=colors[0], linewidth=3)  ### inaccurate
            ax.plot(data_step, data_value, color=colors[0], alpha=0.3)
        axes[-1].set_xlabel('step')

        plt.show()



