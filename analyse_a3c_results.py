import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from xml.etree import ElementTree as ET
from a3c_agent import LOG_PATH, SAVE_PATH, PLOT_PATH, CHECKPOINT

"""Script for analysing agent's actions.

This script reads all XML log files and plots figures and histograms for the rewards, collected minerals and collected
 gas. It finds the best 10 episodes in terms of those three factors and computes statistics about the performed actions. 
"""


def read_xml(filename):
    """Reads and parses and XML file.

    This function reads and parses a given XML file and returns a list of all global episode numbers, rewards, gas and
    XML tags (including actions, if available) of the episodes.

    :param filename: the name of the XML file to read
    :return: a list of all global episode numbers, rewards, gas and XML tags in that XML file
    """
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
    """Computes an averaged mean for a given input.

    :param x: the array with all values
    :param N: the window size for the average
    :return: the averaged mean
    """
    return np.array([np.mean(x[N * i:N * (i + 1)]) for i in range(len(x) // N)])


def best_episodes(results, n=10):
    """Finds the best n episodes for reward, collected minerals and collected gas.

    This function finds the best n episodes for reward, collected minerals and collected gas and prints them to the
    terminal.

    :param results: an array of all episode number, reward, minerals and gas for all episodes
    :param n: how many top episodes the function should return
    :return: a directory with the episode number of the best episodes for reward, minerals and gas
    """
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
    """Analyses the actions performed in the episodes.

    This function iterates over all episodes and if an episode has the performed actions as child nodes, it counts
    whether they were performed because of randomness or not. It prints the frequency of all actions as well as the of
    the random an non-ranodm ones to the terminal.

    :param episodes: a list of XML nodes, representing the episodes
    """
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
    """Main function, it runs the script.
    
    It iterates over all XML files in the LOG_PATH and calls all functions above. It also plots the results to PLOT_PATH.
    """
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
        plt.plot(episodes, v, '.', color='orange', label='Total')
        plt.plot(episodes[::CHECKPOINT], averaged_mean(v, CHECKPOINT), label='Smoothened Average')
        plt.title(k.title())
        plt.legend()

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
