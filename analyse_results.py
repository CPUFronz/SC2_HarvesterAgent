import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET
from a3c_agent import LOG_PATH, SAVE_PATH, PLOT_PATH


def read_xml(filename):
    results = []
    xml = ET.parse(filename)
    root = xml.getroot()

    for episode in root.getchildren():
        num = int(episode.attrib['num_global'])
        reward = float(episode.attrib['reward'])
        minerals = int(episode.attrib['total_collected_minerals'])
        gas = int(episode.attrib['total_collected_gas'])
        results.append((num, reward, minerals, gas))

    return results


def averaged_mean(x, N):
    return np.array([np.mean(x[N * i:N * (i + 1)]) for i in range(len(x) // N)])


def best_episodes(results, n=10):
    top = {
        'rewards': results[results[:, 1].argsort()][::-1,:][:n],
        'minerals': results[results[:, 2].argsort()][::-1,:][:n],
        'gas': results[results[:, 3].argsort()][::-1,:][:n]
    }

    for k, v in top.items():
        print('Top 10 Episodes', k.title())
        print('Episode      Reward   Minerals   Gas')
        for i in v:
            print('{0:>7}   {1:9.7}       {2:4d}   {3:3d}'.format(int(i[0]), i[1], int(i[2]), int(i[3])))

        print()


if __name__ == '__main__':
    with open(SAVE_PATH + 'python_vars.pickle', 'rb') as f:
        python_vars = pickle.load(f)
        step_count = python_vars[0]
        episode_count = python_vars[1]

    results = []
    for i in glob.glob(LOG_PATH + '*.xml'):
        results += read_xml(i)

    results.sort(key=lambda l:l[0])
    results = results[:episode_count] # clipping to only feature
    results = np.array(results)

    result_dict = {}
    episodes = results[:, 0]
    result_dict['rewards'] = results[:, 1]
    result_dict['minerals'] = results[:, 2]
    result_dict['gas'] = results[:, 3]

    if not os.path.exists(PLOT_PATH):
        os.makedirs(PLOT_PATH)

    for k,v in result_dict.items():
        plt.plot(episodes[::1000], averaged_mean(v, 1000))
        plt.title(k.title())
        if k == 'rewards':
            y_label = 'Gained ' + k.title()
        else:
            y_label = 'Collected ' + k.title()

        plt.ylabel(y_label)
        plt.xlabel('Episodes')
        plt.savefig(PLOT_PATH + k + '.png')
        plt.show()
        plt.close()

        plt.hist(v)
        plt.title('Histogram of ' + k.title())
        plt.xlabel(k.title() + ' per Episode')
        plt.ylabel('Episode Frequency')
        plt.savefig(PLOT_PATH + k + '_histogram.png')
        plt.show()
        plt.close()

    best_episodes(results)
