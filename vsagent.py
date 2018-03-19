#!/usr/bin/python3
"""
@author: Franz Papst
"""

import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions, features

# Functions
BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
NOOP = actions.FUNCTIONS.no_op.id
SELECT_POINT = actions.FUNCTIONS.select_point.id
SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
SELECT_UNIT = actions.FUNCTIONS.select_unit.id
GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
ATTACK = actions.FUNCTIONS.Attack_screen.id
MOVE = actions.FUNCTIONS.Move_screen.id

# Features
PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
IDLE_WORKER_COUNT = 7
MINERAL_COUNT = 1

# Unit IDs
TERRAN_COMMANDCENTER = 18
TERRAN_SCV = 45
MINERALFIELD = 341
GEYSER = 342

# Parameters
SCREEN = [0]

VS_SCREEN_SIZE_X = 84
VS_SCREEN_SIZE_Y = 84

class VerySimpleAgent(base_agent.BaseAgent):
    """ A very simple scripted pysc2-agent that collects resources.

    This implementation uses absolute coordinates (more or less), thus it will only work when run on the map
    "CollectMineralsAndGas" from the mini game maps. It can be found here:
    https://github.com/deepmind/pysc2/releases/download/v1.2/mini_games.zip

    It is based on the following tutorial: https://chatbotslife.com/building-a-basic-pysc2-agent-b109cde1477c
    """

    def __init__(self):
        """Initialises the instance."""
        super(VerySimpleAgent, self).__init__()
        self.scv_selected = False
        self.refinery_count = 0
        # only use geysers on the top left and bottom right corner
        self.geyser_locations = [
            (0, VS_SCREEN_SIZE_Y // 2, 0, VS_SCREEN_SIZE_X // 2),
            (VS_SCREEN_SIZE_Y // 2, VS_SCREEN_SIZE_Y, VS_SCREEN_SIZE_X // 2, VS_SCREEN_SIZE_X)]

    def action(self, action_id, parameters):
        """Helper method to execute an action.

        :param action_id: action to be executed
        :param parameters: parameter(s) for that action
        """
        return actions.FunctionCall(action_id, parameters)

    def step(self, obs):
        """One step of the game iteration.

        This method performs all the actions the agent is capable of doing: selecting idle workers and send them to
        gather minerals or building a refinery to collect gas.

        :param obs: the observation of the game state
        :return: the action the agent is going to execute
        """
        super(VerySimpleAgent, self).step(obs)
        unit_type = obs.observation['screen'][UNIT_TYPE]

        if not self.scv_selected:
            if SELECT_IDLE_WORKER in obs.observation['available_actions']:
                self.scv_selected = True
                return self.action(SELECT_IDLE_WORKER, [SCREEN])
        else:
            if obs.observation['player'][IDLE_WORKER_COUNT] > len(self.geyser_locations):
                if GATHER in obs.observation['available_actions']:
                    self.scv_selected = False
                    mineral = (unit_type == MINERALFIELD)
                    mineral_y, mineral_x = mineral.nonzero()
                    return self.action(GATHER, [SCREEN, [mineral_x[10],  mineral_y[10]]])
            else:
                if BUILD_REFINERY in obs.observation['available_actions']:
                    if self.refinery_count < len(self.geyser_locations):
                        self.scv_selected = False
                        g = self.geyser_locations[self.refinery_count]
                        vespine = (unit_type == GEYSER)
                        vespine_y, vespine_x = vespine[g[0]:g[1], g[2]:g[3]].nonzero()
                        vespine_x += g[0]
                        vespine_y += g[2]
                        self.refinery_count += 1
                        target = [np.median(vespine_x), np.median(vespine_y)]
                        return self.action(BUILD_REFINERY, [SCREEN, target])

        return self.action(NOOP, [])