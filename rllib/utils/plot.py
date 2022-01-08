
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob


def generate_args():
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-f', '--file-dir', default='./', type=str, help='..')

    args = argparser.parse_args()
    return args



'''
https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar
https://stats.stackexchange.com/questions/111851/standard-deviation-of-an-exponentially-weighted-mean
https://en.wikipedia.org/wiki/Moving_average#Exponentially_weighted_moving_variance_and_standard_deviation
https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
'''


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
    return mean, variance



colors = ['cornflowerblue', 'limegreen', 'darkorange']
args = generate_args()
# file_paths = sorted(glob.glob(join(args.file_dir, '*.txt')))
file_paths = sorted(glob.glob(join(args.file_dir, '*')))
print('path: ', file_paths)


def plot_reward(reward, idx, label):
    

    mean, variance = smooth(reward, 0.001)
    mean = np.array(mean)
    variance = np.array(variance)
    std = np.sqrt(variance)

    # import pdb; pdb.set_trace()
    print(label)

    plt.plot(mean, color=colors[idx], label=label, linewidth=3)
    # plt.fill_between(np.arange(len(reward)), mean-std, mean+std, interpolate=True, color=colors[idx], alpha=0.3)
    plt.plot(reward, color=colors[idx], alpha=0.3)
    # plt.plot(variance, color=colors[idx], alpha=0.3)

    plt.legend()
    plt.pause(0.001)
    plt.plot(block=False)





plt.figure(num='Rewards')
plt.clf()
plt.title('Total reward')
plt.xlabel('Episode')
plt.ylabel('Reward')


for idx, file_path in enumerate(file_paths):
    aa = np.loadtxt(file_path)
    if len(aa.shape) == 1:
        aa = np.expand_dims(aa, axis=1)
    print(aa.shape)

    sns.set(palette='muted', color_codes=True)

    # plt.title('seaborn: statistical data visualization')
    # # seaborn.distplot(aa[:,1], kde=False, color="b")
    # sns.lineplot(x=aa[:,0], y=aa[:,1], data=aa)

    # import pdb; pdb.set_trace()

    plot_reward(aa[:,-1], idx % len(colors), file_path)


plt.show()

