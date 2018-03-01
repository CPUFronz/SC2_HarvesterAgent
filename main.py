#!/usr/bin/python3

import os
import threading
import time
import tensorflow as tf
from absl import app
from pysc2.env import run_loop
from pysc2.env import sc2_env

from a3c_agent import A3CAgent
from constants import SCREEN_SIZE_X, SCREEN_SIZE_Y, MINIMAP_SIZE_X, MINIMAP_SIZE_Y, PARALLEL_THREADS, SAVE_PATH, LOG_PATH, SHOW


def run_thread(agent, display=False):
    with sc2_env.SC2Env(map_name='CollectMineralsAndGas',
                        agent_race='T',
                        difficulty=None,
                        step_mul=8,
                        game_steps_per_episode=0,
                        screen_size_px=(SCREEN_SIZE_X, SCREEN_SIZE_Y),
                        minimap_size_px=(MINIMAP_SIZE_X, MINIMAP_SIZE_Y),
                        visualize=display) as env:

        run_loop.run_loop([agent], env)


def main(argv):
    summary_writer = tf.summary.FileWriter(LOG_PATH)

    with tf.Session() as session:
        agents = []
        for i in range(PARALLEL_THREADS):
            agent = A3CAgent(session, i, summary_writer)
            if os.path.exists(SAVE_PATH):
                agent.load_checkpoint()

            agents.append(agent)
        agent.initialize()

        threads = []
        show = SHOW
        for agent in agents:
            t = threading.Thread(target=run_thread, args=(agent, show))
            threads.append(t)
            t.start()
            time.sleep(5)
            show = False

        for t in threads:
            t.join()

if __name__ == '__main__':
    print('Starting...')
    app.run(main)

# unit numbers = /home/franz/Schreibtisch/s2client-api/include/sc2api/sc2_typeenums.h
