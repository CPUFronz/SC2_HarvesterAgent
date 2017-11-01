#!/usr/bin/python3

import threading
from absl import app
from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import stopwatch

from vsagent import VerySimpleAgent

PARALLEL_THREADS = 1

def run_thread():
    with sc2_env.SC2Env(map_name='CollectMineralsAndGas',
                        agent_race='T',
                        difficulty=None,
                        step_mul=8,
                        game_steps_per_episode=0,
                        screen_size_px=(84,84),
                        minimap_size_px=(64,64),
                        visualize=True) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)
        agent = VerySimpleAgent()

        run_loop.run_loop([agent], env, 2500)


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