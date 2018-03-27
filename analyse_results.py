import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
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
        results.append((num, reward, minerals, gas, episode))

    return results


def averaged_mean(x, N):
    return np.array([np.mean(x[N * i:N * (i + 1)]) for i in range(len(x) // N)])


def best_episodes(results, n=10):
    top = {
        'rewards': results[results[:, 1].argsort()][::-1,:][:n],
        'minerals': results[results[:, 2].argsort()][::-1,:][:n],
        'gas': results[results[:, 3].argsort()][::-1,:][:n]
    }
    episode_dict = {}.fromkeys(top, [])

    for k, v in top.items():
        print('Top 10 Episodes', k.title())
        print('Episode      Reward   Minerals   Gas')
        for i in v:
            print('{0:>7}   {1:9.7}       {2:4d}   {3:3d}'.format(int(i[0]), i[1], int(i[2]), int(i[3])))
            episode_dict[k].append(int(i[0]))

        print()

    return episode_dict


def analyse_actions(episodes):
    actions_counter = Counter()
    actions_counter_random = Counter()

    for i in episodes:
        for j in i.getchildren():
            random_action = j.attrib['random_action'] == 'True'
            random_postion = j.attrib['random_position'] == 'True'

            action = j.attrib['name']
            actions_counter[action] += 1
            actions_counter_random[(action, random_action)] += 1

    spatial_actions = ('Harvest_Gather_screen', 'move_camera', 'select_point', 'Move_minimap', 'Move_screen',
                       'Build_Refinery_screen', 'Build_SupplyDepot_screen', 'Build_CommandCenter_screen',
                       'Rally_Workers_screen', 'Rally_Workers_minimap')
    non_spatial_actions = ('no_op', 'select_idle_worker', 'Harvest_Return_quick',
                           'Morph_SupplyDepot_Lower_quick', 'Morph_SupplyDepot_Raise_quick')

    total_actions_num = sum(actions_counter.values())
    print('Frequency of Actions')
    for i in actions_counter.most_common():
        print('{0:31s}: {1:3.5}%'.format(i[0], (i[1]/total_actions_num) * 100))
    print()

    print('Frequency of random Actions')
    for i in actions_counter_random.most_common():
        if i[0][1]:
            print('{0:31s}: {1:3.5}%'.format(i[0][0], (i[1] / total_actions_num) * 100))
    print()

    print('Frequency of non random Actions')
    for i in actions_counter_random.most_common():
        if not i[0][1]:
            print('{0:31s}: {1:3.5}%'.format(i[0][0], (i[1] / total_actions_num) * 100))


if __name__ == '__main__':
    with open(SAVE_PATH + 'python_vars.pickle', 'rb') as f:
        python_vars = pickle.load(f)
        step_count = python_vars[0]
        episode_count = python_vars[1]

    print('Results from {:d} episodes:\n'.format(episode_count))

    results = []
    for i in glob.glob(LOG_PATH + '*.xml'):
        results += read_xml(i)

    results.sort(key=lambda l:l[0])
    results = results[:episode_count] # clipping number of episodes to number from pickle file
    results_array = np.array(list(zip(*results))[:4]).transpose()
    results_xml = list(zip(*results))[-1]

    result_dict = {}
    episodes = results_array[:, 0]
    result_dict['rewards'] = results_array[:, 1]
    result_dict['minerals'] = results_array[:, 2]
    result_dict['gas'] = results_array[:, 3]

    if not os.path.exists(PLOT_PATH):
        os.makedirs(PLOT_PATH)

    for k,v in result_dict.items():
        plt.plot(episodes, v, '.', color='orange')
        plt.plot(episodes[::500], averaged_mean(v, 500))
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

    best = best_episodes(results_array)
    analyse_actions(results_xml)
