#!/usr/bin/python3
"""
@author: Franz Papst
"""
import tensorflow as tf
import numpy as np
from collections import namedtuple
from pysc2.lib import actions

from constants import SCREEN_SIZE_X, SCREEN_SIZE_Y, MINIMAP_SIZE_X, MINIMAP_SIZE_Y

# NUM_ACTIONS = len(actions.FUNCTIONS._func_list)
NUM_ACTIONS = 15
MINIMAP_FEAURES = 2
SCREEN_FEATURES = 2
NUM_FEATURES = SCREEN_SIZE_X * SCREEN_SIZE_Y * SCREEN_FEATURES + MINIMAP_SIZE_X * MINIMAP_SIZE_Y * MINIMAP_FEAURES + 6 + 6 + NUM_ACTIONS


class NeuralNetwork:
    def __init__(self, network_type, n_neurons=2000, n_inputs=NUM_FEATURES, n_outputs=NUM_ACTIONS, weights=None, biases=None):
        with tf.name_scope(network_type):
            if not weights or not biases:
                weights = [
                    tf.random_normal(shape=(n_inputs, n_neurons), mean=0.5, stddev=0.5),
                    tf.random_normal(shape=(n_neurons, n_outputs * 3, ), mean=0.5, stddev=0.5)
                ]

                biases = [
                    tf.random_normal(shape=(n_inputs, ), mean=0.5, stddev=0.5),
                    tf.random_normal(shape=(n_neurons, ), mean=0.5, stddev=0.5),
                    tf.random_normal(shape=(n_outputs, 3), mean=0.5, stddev=0.5)
                ]

            self.x = tf.placeholder(dtype=tf.float32, shape=(None, n_inputs), name='x')
            self.y = tf.placeholder(dtype=tf.float32, shape=(None, n_outputs, 3), name='y')
            self.w1 = tf.Variable(weights[0], name='w1')
            self.w2 = tf.Variable(weights[1], name='w2')
            self.b1 = tf.Variable(biases[0], name='b1')
            self.b2 = tf.Variable(biases[1], name='b2')
            self.b3 = tf.Variable(biases[2], name='b3')

            h1 = tf.sigmoid(tf.add(tf.matmul(tf.add(self.x, self.b1), self.w1), self.b2))
            self.output_layer = tf.tanh(tf.add(tf.reshape(tf.matmul(h1, self.w2), shape=(-1, n_outputs, 3)), self.b3))

            self.init = tf.global_variables_initializer()
            optimizer = tf.train.AdamOptimizer(learning_rate=10e-4)
            loss = tf.reduce_mean(tf.square(self.output_layer - self.y))
            self.update = optimizer.minimize(loss)

    @property
    def weights(self):
        return [self.w1, self.w2]

    @property
    def biases(self):
        return [self.b1, self.b2, self.b3]

"""
class Critic:
    def __init__(self, n_neurons=10000, n_inputs=NUM_FEATURES, n_outputs=NUM_ACTIONS, weights=None, biases=None, neuron_discount=1.3):
        with tf.name_scope('critic'):
            if not weights or not biases:
                weights = [
                    tf.random_normal((n_inputs, n_neurons)),
                    tf.random_normal((n_neurons, int(n_neurons / neuron_discount))),
                    tf.random_normal((int(n_neurons / neuron_discount), n_outputs))
                ]

                biases = [
                    tf.random_normal(shape=(n_inputs, )),
                    tf.random_normal(shape=(n_neurons, )),
                    tf.random_normal(shape=(int(n_neurons / neuron_discount),)),
                    tf.random_normal(shape=(n_outputs, ))
                ]

            self.x = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs], name='x')
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, n_outputs], name='y')
            self.w1 = tf.Variable(weights[0], name='w1')
            self.w2 = tf.Variable(weights[1], name='w2')
            self.w3 = tf.Variable(weights[1], name='w3')
            self.b1 = tf.Variable(biases[0], name='b1')
            self.b2 = tf.Variable(biases[1], name='b2')
            self.b3 = tf.Variable(biases[2], name='b3')
            self.b4 = tf.Variable(biases[3], name='b4')

            h1 = tf.sigmoid(tf.add(tf.matmul(tf.add(self.x, self.b1), self.w1), self.b2))
            h2 = tf.sigmoid(tf.add(tf.matmul(h1, self.w2), self.b3))

            print('CRITIC', self.w3.shape, h1.shape, h2.shape)

            self.output_layer = tf.tanh(tf.add(tf.matmul(h2, self.w3), self.b4))
#            h3 = tf.tanh(tf.matmul(h2))

#            self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')

            self.init = tf.global_variables_initializer()
            self.optimizer = tf.train.GradientDescentOptimizer(10e-3)

    def forwardprop(self, s, a):
        pass

    @property
    def weights(self):
        return [self.w1, self.w2]
"""


class DDPGAgent():
    def __init__(self):
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None

        self.actor = NeuralNetwork('actor')
        self.actor_target = NeuralNetwork('actor_target', weights=self.actor.weights)
        critic_neurons = 4250
        self.critic = NeuralNetwork('critic', n_neurons=critic_neurons, n_inputs=NUM_FEATURES + 3)
        self.critic_target = NeuralNetwork('critic_target', n_neurons=critic_neurons, n_inputs=NUM_FEATURES + 3,
                                           weights=self.critic.weights)
        self.tf_session = tf.Session()
        self.tf_session.run(self.actor.init)
        self.tf_session.run(self.actor_target.init)
        self.tf_session.run(self.critic.init)
        self.tf_session.run(self.critic_target.init)

        self.executable_actions = [0, 1, 2, 6, 44, 79, 91, 264, 269, 318, 319, 331, 332, 343, 344]
        self.previous_action = None
        self.previous_state = None
        self.previous_features_reward = [None] * 8

        self.mini_batch_size = 4
        self.replay_buffer_max = 1000
        self.replay_buffer = []
        self.gamma = 0.99
        self.tau = 0.001

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def reset(self):
        self.episodes += 1

    def step(self, obs):
        self.steps += 1
        self.reward += obs.reward

        # TODO: add exploration noise
        # select an action based on the current state
        state = self.create_input_vector(obs.observation)
        exploration_noise = np.random.random((15,3)) / obs.observation['game_loop']
        action = self.select_action(self.tf_session.run(self.actor.output_layer, {self.actor.x: state.reshape(1, -1)}) + exploration_noise, obs.observation['available_actions'])[0]

        if self.previous_action:
            reward = self.give_reward(obs.observation)
            self.add_to_replay_buffer(self.previous_state, self.previous_action, reward, state)

            # TODO: auf TensorFlow (GPU!) umbauen
            # sample from minibatch and reshape
            samples = self.sample_from_replay_buffer
            sampled_states = np.array([row[0] for row in samples])
            sampled_actions = np.array([row[1] for row in samples])
            sampled_rewards = np.array([row[2] for row in samples])
            actor_target_input = {self.actor_target.x: [row[3] for row in samples]}

            # feed the samples to the target networks to obtain y
            actor_target_prediction = self.select_action(self.tf_session.run(self.actor_target.output_layer, feed_dict=actor_target_input))
            critic_target_input = np.append(actor_target_input.popitem()[1], actor_target_prediction, axis=1)
            y = self.gamma * self.tf_session.run(self.critic_target.output_layer, feed_dict={self.critic_target.x: critic_target_input})
            y = (np.multiply(np.ones_like(y), sampled_rewards[:, np.newaxis, np.newaxis])) + y

            # use the y to update the critic
            critic_inputs = np.append(sampled_states, sampled_actions.reshape(self.mini_batch_size, -1), axis=1)
            self.tf_session.run(self.critic.update, feed_dict={self.critic.x: critic_inputs, self.critic.y: y})

        self.previous_action = action
        self.previous_state = state
        return self.execute_action(action)

    def create_input_vector(self, observation):
        state_vector = np.array([])
        state_vector = np.append(state_vector, observation['minimap'][1].flatten())
        state_vector = np.append(state_vector, observation['minimap'][3].flatten())
        state_vector = np.append(state_vector, observation['screen'][5].flatten())
        state_vector = np.append(state_vector, observation['screen'][6].flatten())
        state_vector = np.append(state_vector, observation['player'][1].flatten())
        state_vector = np.append(state_vector, observation['player'][2].flatten())
        state_vector = np.append(state_vector, observation['player'][3].flatten())
        state_vector = np.append(state_vector, observation['player'][4].flatten())
        state_vector = np.append(state_vector, observation['player'][6].flatten())
        state_vector = np.append(state_vector, observation['player'][7].flatten())
        state_vector = np.append(state_vector, observation['score_cumulative'][0].flatten())
        state_vector = np.append(state_vector, observation['score_cumulative'][2].flatten())
        state_vector = np.append(state_vector, observation['score_cumulative'][7].flatten())
        state_vector = np.append(state_vector, observation['score_cumulative'][8].flatten())
        state_vector = np.append(state_vector, observation['score_cumulative'][9].flatten())
        state_vector = np.append(state_vector, observation['score_cumulative'][10].flatten())
        actions = [1 if i in observation['available_actions'] else 0 for i in self.executable_actions]
        state_vector = np.append(state_vector, actions)

        return state_vector

    def add_to_replay_buffer(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))
        if len(self.replay_buffer) > self.replay_buffer_max:
            self.replay_buffer.pop(0)

    @property
    def sample_from_replay_buffer(self):
        return [self.replay_buffer[i] for i in np.random.choice(len(self.replay_buffer), self.mini_batch_size)]

    def select_action(self, action, observation=None):
        selected_actions = []
        for i in range(action.shape[0]):
            action_id_simple = np.argmax(action[i,:,:], axis=0)[0]
            action_id = self.executable_actions[action_id_simple]

            if observation is not None:
                if not action_id in observation:
                    action_id = 0

            pos_arg = True in [True if j.name == 'minimap' or j.name == 'screen' else False for j in self.action_spec.functions[action_id].args]
            if pos_arg:
                x_max = self.action_spec.functions[action_id].args[-1].sizes[0]
                y_max = self.action_spec.functions[action_id].args[-1].sizes[1]
                x = action[i, action_id_simple, 1]
                x = int(x) if x <= x_max else x_max
                y = action[i, action_id_simple, 2]
                y = int(y) if y <= y_max else y_max
                selected_actions.append([action_id, x, y])
            else:
                selected_actions.append([action_id, 0, 0])
        return selected_actions

    def execute_action(self, action):
        # note: actions always get executed immediately, no actions are queued
        action_id = action[0]

        arguments = []

        for arg in self.action_spec.functions[action_id].args:
            if arg.name == 'queued':
                arguments.append([0])
            elif arg.name == 'minimap' or arg.name == 'screen':
                arg_list = [action[1], action[2]]
                arguments.append(arg_list)
            else:
                arguments.append([0])

        return actions.FunctionCall(action_id, arguments)

    def give_reward(self, observation):
        # TODO add reward for building a refinery close to a geyser

        num_workers = observation['player'][6]
        num_idle_workers = observation['player'][7]
        score = observation['score_cumulative'][0]
        idle_worker_time = observation['score_cumulative'][2]
        collected_minerals = observation['score_cumulative'][7]
        collected_gas = observation['score_cumulative'][8]
        collection_rate_minerals = observation['score_cumulative'][9]
        collection_rate_gas = observation['score_cumulative'][10]

        reward = 0

        if all(i is not None for i in self.previous_features_reward):
            if num_workers > self.previous_features_reward[0]:
                reward += 0.05
            if num_idle_workers < self.previous_features_reward[1]:
                reward += 0.75
            if score > self.previous_features_reward[2]:
                reward += 0.1
            if idle_worker_time < self.previous_features_reward[3]:
                reward += 0.5
            if collected_minerals > self.previous_features_reward[4]:
                reward += 0.25
            if collected_gas > self.previous_features_reward[5]:
                reward += 0.25
            if collection_rate_minerals > self.previous_features_reward[6]:
                reward += 1
            if collection_rate_gas > self.previous_features_reward[7]:
                reward += 1

        self.previous_features_reward[0] = num_workers
        self.previous_features_reward[1] = num_idle_workers
        self.previous_features_reward[2] = score
        self.previous_features_reward[3] = idle_worker_time
        self.previous_features_reward[4] = collected_minerals
        self.previous_features_reward[5] = collected_gas
        self.previous_features_reward[6] = collection_rate_minerals
        self.previous_features_reward[7] = collection_rate_gas

        return reward

if __name__ == '__main__':
    agent = DDPGAgent()

# len(actions.FUNCTIONS._func_list)
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# https://github.com/songrotek/DDPG/