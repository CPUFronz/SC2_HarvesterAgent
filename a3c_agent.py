#!/usr/bin/python3
"""
@author: Franz Papst
"""
import os
import time
import pickle
import threading
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from xml.dom import minidom
from xml.etree import ElementTree as ET
from pysc2.lib import actions
from constants import SCREEN_SIZE_X, SCREEN_SIZE_Y, MINIMAP_SIZE_X, MINIMAP_SIZE_Y, NUM_ACTIONS, MINIMAP_FEATURES\
    , SCREEN_FEATURES, NON_SPATIAL_FEATURES, NUM_BATCHES, MAX_STEPS_TOTAL, EXPLORATION_RATE, DISCOUNT_FACTOR\
    , LEARNING_RATE, SAVE_PATH, LOG_PATH, EPSILON, CHECKPOINT, SHOW_PROGRESS, DETAILED_LOGS


class NeuralNetwork:
    def __init__(self, n_outputs=NUM_ACTIONS):
        assert MINIMAP_SIZE_X == SCREEN_SIZE_X, 'resolution of minimap and screen have to be the same (X-axis)'
        assert MINIMAP_SIZE_Y == SCREEN_SIZE_Y, 'resolution of minimap and screen have to be the same (Y-axis)'

        self.minimap = tf.placeholder(shape=(None, MINIMAP_FEATURES, MINIMAP_SIZE_X, MINIMAP_SIZE_Y), dtype=np.float32, name='minimap')
        self.screen = tf.placeholder(shape=(None, SCREEN_FEATURES, SCREEN_SIZE_X, SCREEN_SIZE_Y), dtype=np.float32, name='screen')
        self.non_spatial_features =tf.placeholder(shape=(None, NON_SPATIAL_FEATURES), dtype=np.float32, name='non_spatial_features')

        minimap_conv1 = layers.conv2d(tf.transpose(self.minimap, [0, 2, 3, 1]), num_outputs=16, kernel_size=5, stride=1,scope='minimap_conv1')
        minimap_conv2 = layers.conv2d(minimap_conv1, num_outputs=32, kernel_size=3, stride=1, scope='minimap_conv2')
        screen_conv1 = layers.conv2d(tf.transpose(self.screen, [0, 2, 3, 1]), num_outputs=16, kernel_size=5, stride=1,scope='screen_conv1')
        screen_conv2 = layers.conv2d(screen_conv1, num_outputs=32, kernel_size=3, stride=1, scope='screen_conv2')
        non_spatial_features = layers.fully_connected(layers.flatten(self.non_spatial_features), num_outputs=256, activation_fn=tf.tanh, scope='non_spatial_features')
        feat_conv = tf.concat([minimap_conv2, screen_conv2], axis=3)
        spatial_action = layers.conv2d(feat_conv, num_outputs=1, kernel_size=1, stride=1, activation_fn=None, scope='spatial_action')
        self.spatial_action = tf.nn.softmax(layers.flatten(spatial_action))
        full_features = tf.concat([layers.flatten(minimap_conv2), layers.flatten(screen_conv2), non_spatial_features], axis=1)
        full_features = layers.fully_connected(full_features, num_outputs=256, activation_fn=tf.nn.relu, scope='full_features')
        self.non_spatial_action = layers.fully_connected(full_features, num_outputs=NUM_ACTIONS, activation_fn=tf.nn.softmax, scope='non_spatial_action')
        self.value = tf.reshape(layers.fully_connected(full_features, num_outputs=1, activation_fn=None, scope='value'), [-1])


class A3CAgent():
    step_counter = 0
    episode_counter = 0
    action_logs = {}
    lock_step = threading.Lock()
    lock_episode = threading.Lock()

    def __init__(self, session, agent_id, summary_writer, name='A3CAgent'):
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.episode_start = 0

        self.agent_id = agent_id
        A3CAgent.action_logs[self.agent_id] = None
        reuse = self.agent_id > 0

        self.epsilon = EPSILON
        self.exploration_rate = EXPLORATION_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.executable_actions = [0, 1, 2, 6, 44, 79, 91, 264, 269, 318, 319, 331, 332, 343, 344]
        self.replay_states = []
        self.replay_actions = []

        self.summary = []
        self.summary_writer = summary_writer

        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            self.nn = NeuralNetwork()

            self.valid_spatial_action = tf.placeholder(tf.float32, [None, ], name='valid_spatial_action')
            self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, NUM_ACTIONS], name='valid_non_spatial_action')
            self.spatial_action_selected = tf.placeholder(tf.float32, [None, SCREEN_SIZE_X * SCREEN_SIZE_Y], name='spatial_action_selected')
            self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, NUM_ACTIONS], name='non_spatial_action_selected')
            self.R = tf.placeholder(tf.float32, [None], name='R')

            spatial_action_prob = tf.reduce_sum(tf.multiply(self.nn.spatial_action, self.spatial_action_selected)) # axis=1?
            spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1))

            non_spatial_action_prob = tf.reduce_sum(tf.multiply(self.nn.non_spatial_action, self.non_spatial_action_selected))
            valid_non_spatial_action_prob = tf.reduce_sum(tf.multiply(self.nn.non_spatial_action, self.valid_non_spatial_action))
            valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1)
            non_spatial_action_prob = tf.div(non_spatial_action_prob, valid_non_spatial_action_prob)
            non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1))

            self.summary.append(tf.summary.histogram('spatial_action_prob', spatial_action_prob))
            self.summary.append(tf.summary.histogram('non_spatial_action_prob', non_spatial_action_prob))

            action_log_prob = tf.add(tf.multiply(self.valid_spatial_action, spatial_action_log_prob), non_spatial_action_log_prob)
            advantage = tf.stop_gradient(tf.subtract(self.R, self.nn.value))
            self.policy_loss = -tf.reduce_mean(tf.multiply(action_log_prob, advantage))
            self.value_loss = -tf.reduce_mean(tf.multiply(self.nn.value, advantage))

            loss = tf.add(self.policy_loss, self.value_loss)

            self.summary.append(tf.summary.scalar('policy_loss', self.policy_loss))
            self.summary.append(tf.summary.scalar('value_loss', self.value_loss))

            self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)
            gradients = optimizer.compute_gradients(loss)
            clipped_gradients = []
            for grad, var in gradients:
                self.summary.append(tf.summary.histogram(var.op.name, var))
                self.summary.append(tf.summary.histogram(var.op.name + '/grad', grad))

                grad = tf.clip_by_norm(grad, 10.0)
                clipped_gradients.append([grad, var])
            self.train = optimizer.apply_gradients(clipped_gradients)
            self.summary_op = tf.summary.merge(self.summary)

        self.tf_session = session
        self.saver = tf.train.Saver()

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def reset(self):
        if len(self.replay_states) > 0:
            self.episodes += 1
            self.steps = 0
            with A3CAgent.lock_episode:
                A3CAgent.episode_counter += 1
                global_episode = A3CAgent.episode_counter
                global_steps = A3CAgent.step_counter

            learning_rate = LEARNING_RATE * (1 - 0.9 * A3CAgent.step_counter / MAX_STEPS_TOTAL)
            reward, policy_loss, value_loss = self.update(learning_rate)

            self.save_action_log(global_episode, reward, policy_loss, value_loss)
            self.replay_states = []
            self.replay_actions = []

            if SHOW_PROGRESS:
                print('Episode {0:d} finished, took: {1:4.3f} seconds'.format(global_episode, time.time() - self.episode_start))
            self.episode_start = time.time()

            if global_episode % CHECKPOINT == 0:
                print('Episode: {0:d}, step {1:d}/{2:d}, saving model...'.format(global_episode, global_steps, MAX_STEPS_TOTAL))
                self.save_checkpoint(global_steps, global_episode)
        else:
            self.episode_start = time.time()

    def initialize(self):
        self.tf_session.run(tf.global_variables_initializer())

    def step(self, obs):
        if A3CAgent.step_counter >= MAX_STEPS_TOTAL:
            self.update(LEARNING_RATE)
            # stopping the execution of the threads via an exception
            raise KeyboardInterrupt

        nn_input = self.create_feed_dict(obs.observation)
        non_spatial_action, spatial_action = self.tf_session.run([self.nn.non_spatial_action, self.nn.spatial_action], feed_dict=nn_input)

        available_actions = obs.observation['available_actions']
        valid_actions = set(available_actions).intersection(self.executable_actions)
        valid_actions_mask = np.array([True] * len(self.executable_actions))
        for i in available_actions:
            if i in valid_actions:
                valid_actions_mask[self.executable_actions.index(i)] = False
        non_spatial_action = non_spatial_action.flatten()
        action_id = self.executable_actions[np.argmax(np.ma.array(non_spatial_action, mask=valid_actions_mask))]

        action_target = np.argmax(spatial_action.ravel())
        action_target = (action_target // SCREEN_SIZE_Y, action_target % SCREEN_SIZE_X)

        random_action = False
        random_position = False

        # exploration is done via a combination of epsilon greedy and and adaptive exploration rate
        explore = (A3CAgent.step_counter + ((1 - self.exploration_rate) * MAX_STEPS_TOTAL)) / MAX_STEPS_TOTAL

        if np.random.rand() > explore or np.random.rand() < self.epsilon:
            valid_actions = np.array(list(valid_actions), dtype=np.int32)
            action_id = np.random.choice(valid_actions)
            random_action = True

        if np.random.rand() > explore or np.random.rand() < self.epsilon:
            action_target = (np.random.randint(0, SCREEN_SIZE_Y - 1), np.random.randint(0, SCREEN_SIZE_X - 1))
            random_position = True

        self.replay_states.append((nn_input[self.nn.minimap],
                                   nn_input[self.nn.screen],
                                   nn_input[self.nn.non_spatial_features],
                                   obs.last()))
        self.replay_actions.append((action_id, action_target, list(valid_actions), random_action, random_position))

        arguments = []
        for arg in actions.FUNCTIONS[action_id].args:
            # if the action needs a target, note that select_rect is not supported yet, so only those two are checked
            if arg.name in ('screen', 'minimap'):
                arguments.append(action_target)
            else:
                arguments.append([0])  # only executing direct action, no queuing

        with A3CAgent.lock_step:
            A3CAgent.step_counter += 1

        return actions.FunctionCall(action_id, arguments)

    def update(self, learning_rate):
        # if the last state in the buffer is a terminal state, set R=0
        if self.replay_states[-1][-1]:
            R = 0
        else:
            minimap, screen, non_spatial_features, _ = self.replay_states[-1]
            feed_dict = {self.nn.minimap: minimap,
                         self.nn.screen: screen,
                         self.nn.non_spatial_features: non_spatial_features}
            R = self.tf_session.run(self.nn.value, feed_dict=feed_dict)[0]

        cumulated_rewards = np.zeros(shape=len(self.replay_states,), dtype=np.float32)
        cumulated_rewards[0] = R

        valid_spatial_action = np.zeros(shape=(len(self.replay_states,)), dtype=np.float32)
        spatial_action_selected = np.zeros(shape=(len(self.replay_states), SCREEN_SIZE_X * SCREEN_SIZE_Y), dtype=np.float32)
        valid_non_spatial_action = np.zeros([len(self.replay_states), len(self.executable_actions,)], dtype=np.float32)
        non_spatial_action_selected = np.zeros([len(self.replay_states), len(self.executable_actions)], dtype=np.float32)

        self.replay_states.reverse()
        self.replay_actions.reverse()

        minimap = []
        screen = []
        non_spatial_features = []

        for i in range(len(self.replay_states)):
            mm, scr, info, _ = self.replay_states[i]
            minimap.append(mm)
            screen.append(scr)
            non_spatial_features.append(info)

            # reward is minerals, gas * 10, collection_rate_minerals * 10, collection_rate_gas * 100
            reward = info.flatten()[8] + info.flatten()[9] * 10 + info.flatten()[10] * 10 + info.flatten()[11] * 100

            if i > 0:
                cumulated_rewards[i] = reward + self.discount_factor * cumulated_rewards[i-1]

            action_id, action_target, valid_actions, _ , _ = self.replay_actions[i]
            valid_actions_indices = [0] * len(self.executable_actions)
            for j in valid_actions:
                valid_actions_indices[self.executable_actions.index(j)] = 1

            non_spatial_action_selected[i, self.executable_actions.index(action_id)] = 1

            args = actions.FUNCTIONS[action_id].args
            for arg in args:
                if arg.name in ('screen', 'minimap'):
                    valid_spatial_action[i] = 1
                    index = action_target[1] * SCREEN_SIZE_Y + action_target[0]
                    spatial_action_selected[i, index] = 1

        total_reward = cumulated_rewards[-1]

        minimap = np.array(minimap).squeeze()
        screen = np.array(screen).squeeze()
        non_spatial_features = np.array(non_spatial_features).squeeze()
        non_spatial_action_selected = np.array(non_spatial_action_selected)

        # split the input into batches, to not consume all the GPU memory
        minimap = np.array_split(minimap, NUM_BATCHES)
        screen = np.array_split(screen, NUM_BATCHES)
        non_spatial_features = np.array_split(non_spatial_features, NUM_BATCHES)
        cumulated_rewards = np.array_split(cumulated_rewards, NUM_BATCHES)
        valid_spatial_action = np.array_split(valid_spatial_action, NUM_BATCHES)
        spatial_action_selected = np.array_split(spatial_action_selected, NUM_BATCHES)
        valid_non_spatial_action = np.array_split(valid_non_spatial_action, NUM_BATCHES)
        non_spatial_action_selected = np.array_split(non_spatial_action_selected, NUM_BATCHES)

        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        losses = np.array([], dtype=np.float32).reshape(0,2)

        for i in range(len(minimap)):
            feed_dict = {self.nn.minimap: minimap[i],
                         self.nn.screen: screen[i],
                         self.nn.non_spatial_features: non_spatial_features[i],
                         self.R: cumulated_rewards[i],
                         self.valid_spatial_action: valid_spatial_action[i],
                         self.spatial_action_selected: spatial_action_selected[i],
                         self.valid_non_spatial_action: valid_non_spatial_action[i],
                         self.non_spatial_action_selected: non_spatial_action_selected[i],
                         self.learning_rate: learning_rate}
            _, summary, policy_loss, value_loss = self.tf_session.run([self.train, self.summary_op, self.policy_loss, self.value_loss], feed_dict=feed_dict, options=run_options)
            self.summary_writer.add_summary(summary, A3CAgent.step_counter)

            losses = np.vstack((losses, (policy_loss, value_loss)))

        # reverse it again, so it is in the original order, both lists are used later on
        self.replay_states.reverse()
        self.replay_actions.reverse()

        avg_losses = losses.mean(axis=0)
        return total_reward, avg_losses[0], avg_losses[1]

    def create_feed_dict(self, observation):
        minimap = np.array(observation['minimap'], dtype=np.float32)
        minimap = np.delete(minimap, [0, 2, 4, 5, 6], 0)
        minimap = np.expand_dims(minimap, axis=0)
        screen = np.array(observation['screen'], dtype=np.float32)
        screen = np.delete(screen, [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16], 0)
        screen = np.expand_dims(screen, axis=0)
        # TODO: add available actions as well
        non_spatial_features = np.array([
            observation['player'][1],
            observation['player'][2],
            observation['player'][3],
            observation['player'][4],
            observation['player'][6],
            observation['player'][7],
            observation['score_cumulative'][0],
            observation['score_cumulative'][2],
            observation['score_cumulative'][7],
            observation['score_cumulative'][8],
            observation['score_cumulative'][9],
            observation['score_cumulative'][10]
        ], dtype=np.float32)
        non_spatial_features = np.expand_dims(non_spatial_features, axis=0)

        feed_dict = {self.nn.minimap: minimap,
                     self.nn.screen: screen,
                     self.nn.non_spatial_features: non_spatial_features}
        return feed_dict

    def save_checkpoint(self, global_steps, global_episodes):
        action_logs = A3CAgent.action_logs

        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)

        with open(SAVE_PATH + 'python_vars.pickle', 'wb') as f:
            pickle.dump((global_steps, global_episodes, SCREEN_SIZE_Y, SCREEN_SIZE_X), f)
        self.saver.save(self.tf_session, SAVE_PATH + 'SC2_A3C_harvester.ckpt')

        for k, v in action_logs.items():
            filename = LOG_PATH + 'agent{:02d}.xml'.format(k)
            with open(filename, 'w') as f:
                reparsed = minidom.parseString(ET.tostring(v.getroot()))
                reparsed.writexml(f, addindent='  ', newl='\n')

    def load_checkpoint(self):
        if not os.path.exists(SAVE_PATH):
            raise FileNotFoundError('Could not find saved model.')

        with open(SAVE_PATH + 'python_vars.pickle', 'rb') as f:
            python_vars = pickle.load(f)
            A3CAgent.step_counter = python_vars[0]
            A3CAgent.episode_counter = python_vars[1]
            screen_x = python_vars[2]
            screen_y = python_vars[3]

            assert screen_x == SCREEN_SIZE_X, 'Agent was trained for a different resolution (X-axis)'
            assert screen_y == SCREEN_SIZE_Y, 'Agent was trained for a different resolution (Y-axis)'

        A3CAgent.action_logs[self.agent_id] = ET.parse(LOG_PATH + 'agent{:02d}.xml'.format(self.agent_id))
        root = A3CAgent.action_logs[self.agent_id].getroot()
        self.episodes = int(root.getchildren()[-1].attrib['num_agent'])

        checkpoint = tf.train.get_checkpoint_state(SAVE_PATH)
        self.saver.restore(self.tf_session, checkpoint.model_checkpoint_path)

    def save_action_log(self, num_episode, reward, policy_loss, value_loss):
        if not A3CAgent.action_logs[self.agent_id]:
            root = ET.Element('action_logs')
            tree = ET.ElementTree(root)
        else:
            tree = A3CAgent.action_logs[self.agent_id]

        total_collected_minerals = int(self.replay_states[-1][2].flatten()[8])
        total_collected_gas = int(self.replay_states[-1][2].flatten()[9])

        minerals_per_episode = [(self.episodes, total_collected_minerals)]
        gas_per_episode = [(self.episodes , total_collected_gas)]
        for t in tree.findall('episode'):
            value = int(t.attrib['total_collected_minerals'])
            episode = int(t.attrib['num_agent'])
            minerals_per_episode.append((episode, value))

            value = int(t.attrib['total_collected_gas'])
            episode = int(t.attrib['num_agent'])
            gas_per_episode.append((episode, value))

        # keep detailed logs only for the 5 best results for minerals and gas
        # add all episodes to the list, sort them and prune the list
        minerals_per_episode.sort(key=lambda tup: tup[1], reverse=True)
        gas_per_episode.sort(key=lambda tup: tup[1], reverse=True)
        top_episodes = set([i[0] for i in minerals_per_episode[:DETAILED_LOGS] + gas_per_episode[:DETAILED_LOGS]])
        other_episodes = set([i[0] for i in minerals_per_episode + gas_per_episode]) - top_episodes

        log_entry = ET.SubElement(tree.getroot(), 'episode')
        log_entry.attrib['num_global'] = str(num_episode)
        log_entry.attrib['num_agent'] = str(self.episodes)
        log_entry.attrib['total_collected_minerals'] = str(total_collected_minerals)
        log_entry.attrib['total_collected_gas'] = str(total_collected_gas)
        log_entry.attrib['loss_actor'] = str(policy_loss)
        log_entry.attrib['loss_critic'] = str(value_loss)
        log_entry.attrib['reward'] = str(reward)

        if self.episodes in top_episodes:
            for i, action in enumerate(self.replay_actions):
                performed_action = ET.SubElement(log_entry, 'action')
                performed_action.attrib['name'] = actions.FUNCTIONS[action[0]].name
                performed_action.attrib['x'] = str(action[1][0])
                performed_action.attrib['y'] = str(action[1][1])
                performed_action.attrib['random_action'] = str(action[3])
                performed_action.attrib['random_position'] = str(action[4])

                collected_minerals = self.replay_states[i][2].flatten()[8]
                collected_gas = self.replay_states[i][2].flatten()[9]

                performed_action.attrib['collected_minerals'] = str(int(collected_minerals))
                performed_action.attrib['collected_gas'] = str(int(collected_gas))

        for e in other_episodes:
            remove_entry = tree.find('.//episode[@num_agent="{:d}"]'.format(e))
            for r in remove_entry.findall('action'):
                remove_entry.remove(r)

        A3CAgent.action_logs[self.agent_id] = tree
        