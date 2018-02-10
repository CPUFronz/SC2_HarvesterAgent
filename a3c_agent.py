#!/usr/bin/python3
"""
@author: Franz Papst
"""
import tflearn
import tensorflow as tf
import numpy as np
import tflearn.layers as tfl
from pysc2.lib import actions
from constants import SCREEN_SIZE_X, SCREEN_SIZE_Y, MINIMAP_SIZE_X, MINIMAP_SIZE_Y, NON_SPATIAL_FEATURES, BATCH_SIZE

NUM_ACTIONS = 15
MINIMAP_FEATURES = 2
SCREEN_FEATURES = 2
NUM_FEATURES = SCREEN_SIZE_X * SCREEN_SIZE_Y * SCREEN_FEATURES + MINIMAP_SIZE_X * MINIMAP_SIZE_Y * MINIMAP_FEATURES + NON_SPATIAL_FEATURES + NUM_ACTIONS


class NeuralNetwork:
    def __init__(self, n_outputs=NUM_ACTIONS):

        self.minimap = tflearn.input_data(shape=(None, 2, MINIMAP_SIZE_X, MINIMAP_SIZE_Y), name='minimap')
        self.screen = tflearn.input_data(shape=(None, 3, SCREEN_SIZE_X, SCREEN_SIZE_Y), name='screen')
        self.non_spatial_features = tflearn.input_data(shape=(None, NON_SPATIAL_FEATURES), name='non_spatial_features')

        minimap_conv = tfl.conv_2d(tf.transpose(self.minimap, [0,2,3,1]), nb_filter=16, filter_size=5, activation='relu', name='minimap_conv1')
        minimap_conv = tfl.conv_2d(minimap_conv, nb_filter=32, filter_size=3, activation='relu', name='minimap_conv2')
        screen_conv = tfl.conv_2d(tf.transpose(self.screen, [0,2,3,1]), nb_filter=16, filter_size=5, activation='linear', name='screen_conv1')
        screen_conv = tfl.conv_2d(screen_conv, nb_filter=32, filter_size=3, activation='relu', name='screen_conv2')
        spatial_concat =  tfl.merge([minimap_conv, screen_conv], mode='concat', axis=3, name='spatial_concat')
        self.spatial_action = tfl.flatten(tfl.conv_2d(spatial_concat, nb_filter=1, filter_size=1, activation='softmax', name='spatial_action'))

        full_features = tfl.merge([tfl.flatten(minimap_conv), tfl.flatten(screen_conv), self.non_spatial_features], mode='concat', axis=1)
        feature_layer = tfl.fully_connected(full_features, n_units=256, activation='relu', name='feature_layer')
        self.non_spatial_action = tfl.fully_connected(feature_layer, n_units=n_outputs, activation='softmax', name='non_spatial_action')

        self.value = tfl.fully_connected(full_features, n_units=1, activation=None, name='value')


class A3CAgent():
    def __init__(self, session, reuse, name='A3CAgent'):
        self.reward = 0
        self.episodes = 0
        self.steps = 0

        self.epsilon = 0.1
        self.executable_actions = [0, 1, 2, 6, 44, 79, 91, 264, 269, 318, 319, 331, 332, 343, 344]
        self.replay_states = []
        self.replay_actions = []

        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            self.nn = NeuralNetwork()

            self.valid_spatial_action = tf.placeholder(tf.float32, [None, ], name='valid_spatial_action')
            self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, NUM_ACTIONS], name='valid_non_spatial_action')
            self.spatial_action_selected = tf.placeholder(tf.float32, [None, SCREEN_SIZE_X * SCREEN_SIZE_Y], name='spatial_action_selected')
            self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, NUM_ACTIONS], name='non_spatial_action_selected')
            self.value_target = tf.placeholder(tf.float32, [None], name='value_target')

            spatial_action_prob = tf.reduce_sum(tf.multiply(self.nn.spatial_action, self.spatial_action_selected)) # axis=1?
            spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1))

            non_spatial_action_prob = tf.reduce_sum(tf.multiply(self.nn.non_spatial_action, self.non_spatial_action_selected))
            valid_non_spatial_action_prob = tf.reduce_sum(tf.multiply(self.nn.non_spatial_action, self.valid_non_spatial_action))
            valid_non_spatial_action_prob = tf.log(tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1))
            non_spatial_action_prob = tf.div(non_spatial_action_prob, valid_non_spatial_action_prob)
            non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1))

            action_log_prob = tf.add(tf.multiply(self.valid_spatial_action, spatial_action_log_prob), non_spatial_action_log_prob)
            advantage = tf.stop_gradient(tf.subtract(self.value_target, self.nn.value))
            policy_loss = -tf.reduce_mean(tf.multiply(action_log_prob, advantage))
            value_loss = -tf.reduce_mean(tf.multiply(self.nn.value, advantage))

            loss = tf.add(policy_loss, value_loss)

            self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)
            gradients = optimizer.compute_gradients(loss)
            clipped_gradients = []
            for grad, var in gradients:
                grad = tf.clip_by_norm(grad, 10.0)
                clipped_gradients.append([grad, var])
            self.train = optimizer.apply_gradients(clipped_gradients)

        self.tf_session = session

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def reset(self):
        self.episodes += 1
        self.steps = 0

        if len(self.replay_states) > 0:
            # TODO: give real values
            self.update(0.1,0.1)

    def initialize(self):
        self.tf_session.run(tf.global_variables_initializer())

    def step(self, obs):
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

        if np.random.rand() < self.epsilon:
            valid_actions = np.array(list(valid_actions), dtype=np.int32)
            action_id = np.random.choice(valid_actions)
        if np.random.rand() < self.epsilon:
            action_target = (np.random.randint(0, SCREEN_SIZE_Y - 1), np.random.randint(0, SCREEN_SIZE_X - 1))

        self.replay_states.append((nn_input[self.nn.minimap],
                                   nn_input[self.nn.screen],
                                   nn_input[self.nn.non_spatial_features],
                                   obs.last()))
        self.replay_actions.append((action_id, action_target, list(valid_actions)))

        arguments = []
        for arg in actions.FUNCTIONS[action_id].args:
            # if the action needs a target, note that select_rect is not supported yet, so only those two are checked
            if arg.name in ('screen', 'minimap'):
                arguments.append(action_target)
            else:
                arguments.append([0])  # only executing direct action, no queuing

        return actions.FunctionCall(action_id, arguments)

    def update(self, discount_factor, learning_rate):
        # if the last state in the buffer is a terminal state, set R=0
        if self.replay_states[-1][-1]:
            R = 0
        else:
            minimap, screen, non_spatial_features, _ = self.replay_states[-1]
            feed_dict = {self.nn.minimap: minimap,
                         self.nn.screen: screen,
                         self.nn.non_spatial_features: non_spatial_features}
            R = self.tf_session.run(self.nn.value, feed_dict=feed_dict)[0]

        value_target = np.zeros(shape=len(self.replay_states,), dtype=np.float32)
        value_target[0] = R

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

            #TODO: more fine grained reward function
            reward = info.flatten()[8] + info.flatten()[9]
            if i > 0:
                value_target[i] = reward + discount_factor * value_target[i-1]

            action_id, action_target, valid_actions = self.replay_actions[i]
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

        minimap = np.array(minimap).squeeze()
        screen = np.array(screen).squeeze()
        non_spatial_features = np.array(non_spatial_features).squeeze()
        non_spatial_action_selected = np.array(non_spatial_action_selected)


        minimap = np.array_split(minimap, BATCH_SIZE)
        screen = np.array_split(screen, BATCH_SIZE)
        non_spatial_features = np.array_split(non_spatial_features, BATCH_SIZE)
        value_target = np.array_split(value_target, BATCH_SIZE)
        valid_spatial_action = np.array_split(valid_spatial_action, BATCH_SIZE)
        spatial_action_selected = np.array_split(spatial_action_selected, BATCH_SIZE)
        valid_non_spatial_action = np.array_split(valid_non_spatial_action, BATCH_SIZE)
        non_spatial_action_selected = np.array_split(non_spatial_action_selected, BATCH_SIZE)

        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        for i in range(len(minimap)):
            feed_dict = {self.nn.minimap: minimap[i],
                         self.nn.screen: screen[i],
                         self.nn.non_spatial_features: non_spatial_features[i],
                         self.value_target: value_target[i],
                         self.valid_spatial_action: valid_spatial_action[i],
                         self.spatial_action_selected: spatial_action_selected[i],
                         self.valid_non_spatial_action: valid_non_spatial_action[i],
                         self.non_spatial_action_selected: non_spatial_action_selected[i],
                         self.learning_rate: learning_rate}
            self.tf_session.run(self.train, feed_dict=feed_dict, options=run_options)

        self.replay_states = []
        self.replay_actions = []

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


if __name__ == '__main__':
    agent = A3CAgent()
    nn = NeuralNetwork()
