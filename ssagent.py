#!/usr/bin/python3
"""
@author: Franz Papst
"""

import random
import numpy as np
from collections import defaultdict

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

NOOP = actions.FUNCTIONS.no_op.id
SELECT_POINT = actions.FUNCTIONS.select_point.id
BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
LOWER_SUPPLY_DEPOT = actions.FUNCTIONS.Morph_SupplyDepot_Lower_quick.id
SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id

PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
PLAYER_ID = features.SCREEN_FEATURES.player_id.index

PLAYER_SELF = 1

TERRAN_COMMAND_CENTER = 18
TERRAN_SCV = 45
TERRAN_SUPPLY_DEPOT = 19
MINERAL_FIELD = 341
GEYSER = 342

SCREEN = [0]

ACTION_DO_NOTHING = 'do_nothing'
ACTION_SELECT_SCV = 'select_scv'
ACTION_BUILD_SUPPLY_DEPOT = 'build_supply_depot'
ACTION_BUILD_REFINERY = 'build_refinery'
ACTION_SELECT_COMMAND_CENTER = 'select_command_center'
ACTION_BUILD_SCV = 'build_scv'
ACTION_GATHER_RESOURCES = 'gather_resources'
ACTION_LOWER_SUPPLY_DEPOT = 'lower_supply_depot'
ACTION_SELECT_IDLE_SCV = 'select_idle_scv'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_SCV,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_REFINERY,
    ACTION_SELECT_COMMAND_CENTER,
    ACTION_BUILD_SCV,
    ACTION_GATHER_RESOURCES,
    ACTION_SELECT_IDLE_SCV
]

REWARD_COLLECT_MINERAL = 0.3
REWARD_COLLECT_GAS = 0.6
REWARD_BUILD_SCV = 0.1
REWARD_SCV_BUSY = 0.1
REWARD_LOWER_SUPPLY_DEPOT = 0.01


class QLearningTable:
    """A very naive implementation of Q-learning using a table.

    In order to avoid extra dependencies a defaultdict that stores a numpy array is used. It has a significantly lower
    performance compared to the implementation (https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow) it
    was based on, which used pandas. For its purpose the performance is still ok.
    """
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_geedy=0.9):
        """Initialises the Q-learning table.

        :param actions: the list of actions to chose from
        :param learning_rate: learning rate of the algorithm
        :param reward_decay: discount rate for future rewards
        :param e_geedy: rate of exploitation vs. exploration
        """
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_geedy
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))

    def choose_action(self, observation):
        """Choose which action to take.

        :param observation: current state of observations (list converted to string)
        :return: selected action
        """
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table[observation]
            action = state_action.argmax()
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        """Update the Q-learning table based on the rewards the agent receives.

        :param s: previous state (list converted to string)
        :param a: previous action
        :param r: reward for previous action
        :param s_: next state (list converted to string)
        """
        q_predict = self.q_table[s][a]
        q_target = r + self.gamma * self.q_table[s_].max()
        self.q_table[s][a] += self.lr * (q_target-q_predict)


class SlightlySmarterAgent(base_agent.BaseAgent):
    """ A slightly smarter pysc2-agent for collecting resources using simple Q-learning

    The implementation will harvest minerals and build refineries on geysers next to the agent's starting position, this
    agent does not explore the map.

    It is based on the following tutorial: https://chatbotslife.com/building-a-smart-pysc2-agent-cdc269cb095d
    """

    def __init__(self):
        """Initialises the instance"""
        super(SlightlySmarterAgent, self).__init__()
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        self.gathered_resources_total = 0
        self.previous_mineral_count = 0
        self.previous_gas_count = 0
        self.previous_scv_count = 12
        self.previous_idle_workers = 12
        self.previous_action = None
        self.previous_state = None

    def action(self, action_id, parameters):
        """Helper method to execute an action.

        :param action_id: action to be executed
        :param parameters: parameter(s) for that action
        """
        return actions.FunctionCall(action_id, parameters)

    def step(self, obs):
        """One step of the game iteration.

        This method performs all the actions the agent is capable of doing: selecting (idle) workers, sending them to
        gather resources, training more SCVs, building a refinery or supply depot or simply doing nothing.

        :param obs: the observation of the game state
        :return: the action the agent is going to execute
        """
        super(SlightlySmarterAgent, self).step(obs)

        unit_type = obs.observation['screen'][UNIT_TYPE]
        mineral_count = obs.observation['player'][1]
        gas_count = obs.observation['player'][2]
        supply_limit = obs.observation['player'][4]
        scv_count = obs.observation['player'][6]
        idle_workers = obs.observation['player'][7]

        current_state = [
            mineral_count,
            gas_count,
            supply_limit,
            scv_count,
            idle_workers
        ]

        reward = 0
        if self.previous_action is not None:
            if mineral_count > self.previous_mineral_count:
                reward += REWARD_COLLECT_MINERAL
                self.gathered_resources_total += (mineral_count-self.previous_mineral_count)

            if gas_count > self.previous_gas_count:
                reward += REWARD_COLLECT_GAS
                self.gathered_resources_total += (gas_count-self.previous_gas_count)

            if scv_count > self.previous_scv_count:
                reward += REWARD_BUILD_SCV

            if idle_workers < self.previous_idle_workers:
                reward += REWARD_SCV_BUSY

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]

        self.previous_gas_count = gas_count
        self.previous_mineral_count = mineral_count
        self.previous_state = current_state
        self.previous_action = rl_action

        if smart_action == ACTION_DO_NOTHING:
            return self.action(NOOP, [])

        elif smart_action == ACTION_SELECT_SCV:
            unit_y, unit_x = (unit_type == TERRAN_SCV).nonzero()
            if unit_y.any():
                i = random.randint(0, len(unit_y) - 1)
                target = [unit_x[i], unit_y[i]]
                return self.action(SELECT_POINT, [SCREEN, target])

        elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            if BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                height_map = obs.observation['screen'][0]
                site_y, site_x = (height_map == 0).nonzero()
                if site_y.any():
                    i = random.randint(0, len(site_y) - 1)
                    target =[site_x[i], site_y[i]]
                    return self.action(BUILD_SUPPLY_DEPOT, [SCREEN, target])

        elif smart_action == ACTION_BUILD_REFINERY:
            if BUILD_REFINERY in obs.observation['available_actions']:
                vespine = (unit_type == GEYSER)
                vespine_y, vespine_x = vespine.nonzero()
                if vespine_y.any():
                    i = random.randint(0, len(vespine_y) - 1)
                    target = [vespine_x[i], vespine_y[i]]
                    print('target verspine: ', vespine_x[i], vespine_y[i])
                    return self.action(BUILD_REFINERY, [SCREEN, target])

        elif smart_action == ACTION_SELECT_COMMAND_CENTER:
            unit_y, unit_x = (unit_type == TERRAN_COMMAND_CENTER).nonzero()
            if unit_y.any():
                target = [unit_x.mean(), unit_y.mean()]
                return self.action(SELECT_POINT, [SCREEN, target])

        elif smart_action == ACTION_GATHER_RESOURCES:
            if GATHER in obs.observation['available_actions']:
                mineral_y, mineral_x = (unit_type == MINERAL_FIELD).nonzero()
                if mineral_y.any():
                    i = random.randint(0, len(mineral_y)-1)
                    target = [mineral_x[i], mineral_y[i]]
                return self.action(GATHER, [SCREEN, target])

        elif smart_action == ACTION_BUILD_SCV:
            if TRAIN_SCV in obs.observation['available_actions']:
                return self.action(TRAIN_SCV, [SCREEN])

        elif smart_action == ACTION_SELECT_IDLE_SCV:
            if SELECT_IDLE_WORKER in obs.observation['available_actions']:
                return self.action(SELECT_IDLE_WORKER, [SCREEN])

        return self.action(NOOP, [])
