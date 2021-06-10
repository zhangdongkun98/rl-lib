import matplotlib.pyplot as plt

with open("reward.txt", 'r') as f:
    data = f.readlines()
    reward_list = []
    for subdata in data:
        reward_list.append(float(subdata))

    xx = range(len(reward_list))
    plt.plot(xx, reward_list)

    plt.show()