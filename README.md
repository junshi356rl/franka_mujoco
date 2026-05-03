# PPO Implementation for Continuous Control (Franka Push)

A robust Proximal Policy Optimization (PPO) implementation tailored for continuous robotic manipulation in MuJoCo + Gym environments. Built on PyTorch and extending the [franka_mujoco](https://github.com/learningLogisticsLab/franka_mujoco) framework, this code enhances vanilla PPO with modern stabilization, normalization, and diagnostic techniques widely adopted in contemporary reinforcement learning research.

## Key Mechanisms Beyond Vanilla PPO

While the core follows the original PPO clipped surrogate objective, this implementation integrates several enhancements that significantly improve sample efficiency, training stability, and convergence behavior:

| Mechanism | Purpose & Effect |
|:---|:---|
| **Generalized Advantage Estimation (GAE)** | Reducegradient variance while maintaining low bias forsmoother policy updates. |
| **Online Observation Normalization** | Dynamically scales heterogeneous state features to zero-mean/unit-variance, preventing gradient instability without manual tuning. |
| **Advantage Normalization & Gradient Clipping** |Standardizes advantage signals and caps gradientmagnitudes to ensure stable, bounded policy updates. |
| **Dense Reward Shaping** | Combines task rewards withcontinuous distance feedback to accelerate explorationby encouraging the gripper to approach the cube. |
| **Tanh-Squashed Policy & Log-Std Clamping** | Boundsaction outputs and restricts policy variance to fit inthe task space. |
| **Linear Learning Rate Decay** | Gradually reduces theoptimizer step size to zero, enabling aggressive earlylearning and stable fine-tuning near convergence. |
| **Entropy Regularization** | Adds an exploration bonus to the policy loss to prevent premature convergence and maintain robust exploration throughout training. |

## Test results:
The algorithm converges to a success rate of approximately 95% after 2.5 million steps. When the Franka arm’s physical limitations prevent the cube from reaching the exact target, the system consistently positions it at the closest feasible location to the goal.

Rollout Visualization:
<img src="./docs/PPO/push_PPO.gif" alt=""/>

## Usage

```
# Install dependencies
pip install torch numpy matplotlib pandas

# Run training (outputs to ppo_push_output/<timestamp>/)
python ppo_push.py

```

## Output Structure

```
ppo_push_output/YYYYMMDD_HHMMSS/
├── batch_log_*.csv          # Per-batch detailed metrics
├── total_batch_log.csv      # Merged training log
├── batch_success_rate.png   # Success rate curve
├── ppo_push_actor.pth       # Trained policy weights
├── ppo_push_critic.pth      # Trained value function weights
└── ppo_push_normalizer.pkl  # Observation normalization state
```

# Overview of panda mujoco gym (The previous README in the base repo)
-------------------------

# Open-Source Reinforcement Learning Environments Implemented in MuJoCo with Franka Manipulator

This repository is inspired by [panda-gym](https://github.com/qgallouedec/panda-gym.git) and [Fetch](https://robotics.farama.org/envs/fetch/) environments and is developed with the Franka Emika Panda arm in [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) on the MuJoCo physics engine. Three open-source environments corresponding to three manipulation tasks, `FrankaPush`, `FrankaSlide`, and `FrankaPickAndPlace`, where each task follows the Multi-Goal Reinforcement Learning framework. DDPG, SAC, and TQC with HER are implemented to validate the feasibility of each environment. Benchmark results are obtained with [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) and shown below.

There is still a lot of work to be done on this repo, so please feel free to raise an issue and share your idea!

## Tasks
<div align="center">

`FrankaPushSparse-v0` | `FrankaSlideSparse-v0` | `FrankaPickAndPlaceSparse-v0`
|:------------------------:|:------------------------:|:------------------------:|
<img src="./docs/push.gif" alt="" width="200"/> | <img src="./docs/slide.gif" alt="" width="200"/> | <img src="./docs/pnp.gif" alt="" width="200"/>
</div>

## Benchmark Results

<div align="center">

`FrankaPushSparse-v0` | `FrankaSlideSparse-v0` | `FrankaPickAndPlaceSparse-v0`
|:------------------------:|:------------------------:|:------------------------:|
<img src="./docs/FrankaPushSparse-v1.jpg" alt="" width="230"/> | <img src="./docs/FrankaSlideSparse-v1.jpg" alt="" width="230"/> | <img src="./docs/FrankaPickSparse-v1.jpg" alt="" width="230"/>

</div>

## Installation
Create a virtual environment for python 3.10:
```
conda create --name fm_env python==3.10
```
### Activate your environment:
```
conda activate fm_env
```

And now set python interpreter paths and install dependencies
```
cd franka_mujoco
pip install -e .
pip install -r requirements.txt
```

### Create an fm_env alias
For convenience, you can create an alias in your .bashrc/.zshrc file to quickly load the environment and cd to a desired folder:
```
gedit ~/.bashrc # or gedit ~/.zshrc

# Go to the last line and type
alias fm_env="conda activate fm_env; cd ~/code/franka_mujoco"
# Save and exit your file

# Activate new .bashrc file by calling the following command in your terminal
source .bashrc # or source .zshrc
```

## Test

```python
import sys
import time
import gymnasium as gym
import panda_mujoco_gym

if __name__ == "__main__":
    env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human")

    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        time.sleep(0.2)

    env.close()

```

## Citation

If you use this repo in your work, please cite:

```
@misc{xu2023opensource,
      title={Open-Source Reinforcement Learning Environments Implemented in MuJoCo with Franka Manipulator}, 
      author={Zichun Xu and Yuntao Li and Xiaohang Yang and Zhiyuan Zhao and Lei Zhuang and Jingdong Zhao},
      year={2023},
      eprint={2312.13788},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
