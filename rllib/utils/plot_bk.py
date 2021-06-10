
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob


def generate_args():
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-f', '--file-dir', default='None', type=str, help='..')

    args = argparser.parse_args()
    return args



'''
https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar
https://stats.stackexchange.com/questions/111851/standard-deviation-of-an-exponentially-weighted-mean
https://en.wikipedia.org/wiki/Moving_average#Exponentially_weighted_moving_variance_and_standard_deviation
https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
'''
from typing import List
def smooth_v1(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def smooth_v2(scalars, weight):
    mean, variance = [], []
    m_old, v_old = scalars[0], scalars[0]
    for s in scalars:
        m = weight * m_old + (1 - weight) * s
        v = (1 - weight) * (v_old + weight * (s - m_old)**2)
        mean.append(m)
        variance.append(v)
        m_old, v_old = m, v
    return mean, variance


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
file_paths = sorted(glob.glob(join(args.file_dir, '*.txt')))
print('path: ', file_paths)

def plot_reward_v1(reward, idx):
    rewards = pd.Series(reward)
    means = rewards.rolling(window=100).mean()
    # mstd = rewards.rolling(window=100).std()



    # plt.plot(rewards, color=colors[idx], alpha=0.3)

    # plt.fill_between(mstd.index, means-2*mstd, means+2*mstd, color=colors[idx], alpha=0.3)
    plt.plot(means, color=colors[idx])
    plt.pause(0.001)
    plt.plot(block=False)


def plot_reward(reward, idx, label):
    

    mean, variance = smooth(reward, 0.001)
    mean = np.array(mean)
    variance = np.array(variance)
    std = np.sqrt(variance)

    # import pdb; pdb.set_trace()
    print(label)

    plt.plot(mean, color=colors[idx], label=label)
    # plt.fill_between(np.arange(len(reward)), mean-std, mean+std, interpolate=True, color=colors[idx], alpha=0.3)
    plt.plot(reward, color=colors[idx], alpha=0.3)
    # plt.plot(variance, color=colors[idx], alpha=0.3)

    plt.legend()
    plt.pause(0.001)
    plt.plot(block=False)


# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# sns.set()


# class RewardViewer(object):
#     def __init__(self):
#         self.rewards = []

#     def update(self, reward):
#         self.rewards.append(reward)
#         # self.display()

#     def display(self):
#         plt.figure(num='Rewards')
#         plt.clf()
#         plt.title('Total reward')
#         plt.xlabel('Episode')
#         plt.ylabel('Reward')

#         rewards = pd.Series(self.rewards)
#         means = rewards.rolling(window=100).mean()
#         plt.plot(rewards)
#         plt.plot(means)
#         plt.pause(0.001)
#         plt.plot(block=False)


# rv = RewardViewer()
# for file_path in file_paths:
#     rv.update(np.loadtxt(file_path)[:,1])

# rv.display()


# exit(0)



plt.figure(num='Rewards')
plt.clf()
plt.title('Total reward')
plt.xlabel('Episode')
plt.ylabel('Reward')


for idx, file_path in enumerate(file_paths):
    aa = np.loadtxt(file_path)
    print(aa.shape)

    sns.set(palette='muted', color_codes=True)

    # plt.title('seaborn: statistical data visualization')
    # # seaborn.distplot(aa[:,1], kde=False, color="b")
    # sns.lineplot(x=aa[:,0], y=aa[:,1], data=aa)

    plot_reward(aa[:,1], idx % len(colors), file_path)


plt.show()

