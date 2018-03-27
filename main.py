#!/usr/bin/python3

import os
import sys
import threading
import time
import tensorflow as tf
from absl import app
from absl import flags
from pysc2.env import run_loop
from pysc2.env import sc2_env

from vsagent import VerySimpleAgent
from vsagent import VS_SCREEN_SIZE_X, VS_SCREEN_SIZE_Y
from ssagent import SlightlySmarterAgent
from a3c_agent import A3CAgent
from a3c_agent import A3C_SCREEN_SIZE_X, A3C_SCREEN_SIZE_Y, A3C_MINIMAP_SIZE_X, A3C_MINIMAP_SIZE_Y, PARALLEL_THREADS, SAVE_PATH,\
    LOG_PATH, RENDER, TRAINING


flags.DEFINE_string("agent", "a3c", "Which agent to run.")

def run_thread(agent, ssize_x, ssize_y, msize_x, msize_y, display=False):
    with sc2_env.SC2Env(map_name='CollectMineralsAndGas',
                        agent_race='T',
                        difficulty=None,
                        step_mul=8,
                        game_steps_per_episode=0,
                        screen_size_px=(ssize_x, ssize_y),
                        minimap_size_px=(msize_x, msize_y),
                        visualize=display) as env:

        run_loop.run_loop([agent], env)


def start_a3c_agent():
    summary_writer = tf.summary.FileWriter(LOG_PATH)

    if not TRAINING:
        parallel = 1
        show_render = True
    else:
        parallel = PARALLEL_THREADS
        show_render = RENDER

    with tf.Session() as session:
        agents = []
        for i in range(parallel):
            agent = A3CAgent(session, i, summary_writer)
            restored_session = False
            if os.path.exists(SAVE_PATH):
                agent.load_checkpoint()
                restored_session = True

            agents.append(agent)

        if not restored_session:
            agent.initialize()

        threads = []
        show = show_render
        for agent in agents:
            thread_args = (agent, A3C_SCREEN_SIZE_X, A3C_SCREEN_SIZE_Y, A3C_MINIMAP_SIZE_X, A3C_MINIMAP_SIZE_Y, show)
            t = threading.Thread(target=run_thread, args=thread_args)
            threads.append(t)
            t.start()
            time.sleep(5)
            show = False

        for t in threads:
            t.join()


def start_simple_agent():
    agent = VerySimpleAgent()
    run_thread(agent, VS_SCREEN_SIZE_X, VS_SCREEN_SIZE_Y, VS_SCREEN_SIZE_X, VS_SCREEN_SIZE_Y, True)


def start_qlearn_agent():
    agent = SlightlySmarterAgent()
    run_thread(agent, VS_SCREEN_SIZE_X, VS_SCREEN_SIZE_Y, VS_SCREEN_SIZE_X, VS_SCREEN_SIZE_Y, True)


def main(argv):
    if flags.FLAGS.agent == 'a3c':
        start_a3c_agent()
    elif flags.FLAGS.agent == 'ssagent':
        start_qlearn_agent()
    elif flags.FLAGS.agent == 'vsagent':
        start_simple_agent()


if __name__ == '__main__':
    print('Starting...')
    app.run(main)
