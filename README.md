# Harvester Agent for pysc2

## About

This is the code for a StarCraft 2 deep reinforcing learning agent that is trained to harvest resources, I did as a semester project. It uses the `CollectMineralsAndGas` mini-game to train the agent. I implemented three different types of agents: a very simple, scripted agent [vsagent.py](./vsagent.py), a slightly smarter, Q-learning based agent [ssagent.py](./ssagent.py) and an A3C agent [a3c_agent.py](./a3c_agent.py). Note that the first two agents were only written for getting familiar with the StarCraft 2 environment, most of the effort went into the A3C agent.
It also contains a little utility for analysing the logfiles created by the A3C agent, [analyse_a3c_results.py](./analyse_a3c_results.py), that analyses the logfiles and creates plots out of them.
If you want to learn more about the StarCraft 2 environment, the agent (it's architcture and the implmentation of the A3C algorithm) or the performance of the agent, check out [the report](./report/report.pdf).

## Installation
The easiest way to install the agent is by running [install.sh](./install.sh) it downloads and unpacks all needed archives, it also sets up a virtual environment and adds the `SC2PATH` environment variable to the bash of the virtual environment. By default it installs the CPU version of TensorFlow, since the installation of the GPU powered TensorFlow can't be automatised (you need to register at NVIDIA in order to download [cuDNN](https://developer.nvidia.com/cudnn), which is required for TensorFlow).

The script takes two paramters: `-dir=<DIR>`and `-gpu`, `-dir` specifies the directory where everything should be installed, if it is not set, the script will install it into the current working directory, `-gpu` specifies if GPU supported TensorFlow should be installed or not, **only** use it if you already set up the required libraries for TensorFlow. An example call for the script looks like this:
`./install.sh -dir=~/workspace/SC2_RL`

### Requirements
The agents were developed using the following libraries:
```
tensorflow-gpu==1.5.0
PySC2==1.2
numpy==1.14.0
matplotlib==2.1.2
```

**Important**: [pysc2 2.0](https://github.com/deepmind/pysc2/releases/tag/v2.0) introduced some breaking changes which this agent is not compatible with.

In order to run the agent you need to have pysc2 installed and configured, as well as StarCraft 2 and the maps of the minigames. Check out the [pysc2 readme](https://github.com/deepmind/pysc2/blob/master/README.md) for more information.
For performance reasons I highly recommend using TensorFlow with GPU support, check out the official TensorFlow page on [how to install it](https://www.tensorflow.org/install/install_linux). The CPU version of course works as well, but the performance is very poor. The agent was tested and optimised to run on a GPU with at least 4GB, it should work with less as well, but if you get a `ResourceExhaustedError` try increasing `NUM_BATCHES` from [a3c_agent.py](./a3c_agent.py).

## Usage
To start the agent simply run the main file `python main.py`. It has one command line parameter, to specify which agent to run, by default it is set to `a3c`. The other options are `vsagent` for a very simple scripted agent or `ssagent` for a slightly smarter Q-learning agent.

Once the A3C agent has produces some log files, you can run the analysing tool via `python analyse_a3c_results.py` to analyse the actions and create plots.

## Acknowledgement
The code for the A3C agent is is based on https://github.com/xhujoy/pysc2-agents
The code for the scripted agent and the Q-learning agent are based on https://github.com/skjb/pysc2-tutorial
