# ATCI-PPO


This is a Python implementation of the PPO algorithm.

PPO is a policy gradient method used in reinforcement learning which innovates by using a novel objective function that enables multiple epochs of minibatch
updates. Unlike traditional policy gradient methods, which perform a single update per data sample, PPO enables a more efficient use of data through its ability to reuse samples for multiple updates, which can lead to faster learning rates and improved stability.

## Run instructions

Create a new conda environment and install the required packages:

```bash
conda create --name atci python=3.9
pip install -r requirements.txt
```

Files must be run from the root directory of the repository. 

## PPO implementation

The PPO implementation is located in `./src/`.
PPO Agent can be found in `./src/PPOAgent.py`

## Experiments

Experiment scripts are located in the `./src/experiments/` folder. They generate result files which are stored in the `./results/` folder by default. 
In this folder, you will find some of the results used to generate the plots and tables in the report. Some of them have not finished their complete execution due to memory crashes, but the results are still available. Also, not all of the results are included in the report.

## Report

The report is located in `./doc/` folder.