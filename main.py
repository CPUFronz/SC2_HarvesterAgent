#!/usr/bin/python3

import threading
from absl import app
from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import stopwatch

from ddpg_agent import DDPGAgent
from rand import RandomAgent
from constants import SCREEN_SIZE_X, SCREEN_SIZE_Y, MINIMAP_SIZE_X, MINIMAP_SIZE_Y

PARALLEL_THREADS = 1

def run_thread():
    with sc2_env.SC2Env(map_name='CollectMineralsAndGas',
                        #map_name='Simple64',
                        agent_race='Z',
                        difficulty=None,
                        step_mul=1,
                        game_steps_per_episode=0,
                        screen_size_px=(SCREEN_SIZE_X,SCREEN_SIZE_Y),
                        minimap_size_px=(MINIMAP_SIZE_X,MINIMAP_SIZE_Y),
                        visualize=True) as env:
#        env = available_actions_printer.AvailableActionsPrinter(env)
        agent = DDPGAgent()

        run_loop.run_loop([agent], env)


def main(argv):
    threads = []
    for _ in range(PARALLEL_THREADS):
        t = threading.Thread(target=run_thread)
        threads.append(t)
        t.start()

if __name__ == '__main__':
    print('Starting...')
    app.run(main)

# unit numbers = /home/franz/Schreibtisch/s2client-api/include/sc2api/sc2_typeenums.h