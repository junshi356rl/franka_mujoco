import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../../")))
from panda_mujoco_gym.envs.push import FrankaPushEnv
from utils import MLP, action_forward, ObsNormalizer


def make_obs(state):
    return np.concatenate([state['observation'], state['desired_goal']])


def evaluate(actor_path,
             reward_type='dense',
             render_mode='human',
             num_episodes=10,
             max_steps=50,
             action_size=2,
             state_size=18 + 3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = FrankaPushEnv(reward_type=reward_type, render_mode=render_mode)
    obs_normalizer = ObsNormalizer(dim=state_size)
    # Load ObsNormalizer state
    normalizer_path = os.path.join(os.path.dirname(actor_path), 'ppo_push_normalizer.pkl')
    if os.path.exists(normalizer_path):
        obs_normalizer = ObsNormalizer.load(normalizer_path)
        print(f"Loaded ObsNormalizer from {normalizer_path}")
    else:
        print(f"Warning: ObsNormalizer not found at {normalizer_path}, using default")
    actor = MLP(state_size, action_size, [128, 64], action_size * 2).to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    episode_results = []
    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        obs_np = make_obs(state)
        obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
        obs_norm = obs_normalizer.normalize(obs)

        episode_reward = 0.0
        success = False
        for step in range(max_steps):
            with torch.no_grad():
                actor_out = action_forward(actor, obs_norm)
                action_mean = actor_out[:action_size]

            action = action_mean.cpu().numpy()
            xyz_action = np.append(action, 0.03)
            state, reward, terminated, truncated, info = env.step(xyz_action)
            episode_reward += reward
            success = info.get('is_success', False) or success

            obs_np = make_obs(state)
            obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
            obs_norm = obs_normalizer.normalize(obs)

            if render_mode is not None:
                env.render()
            if terminated or truncated:
                break

        episode_results.append((episode_reward, success, step + 1))
        print(f"Episode {ep:02d} | Reward {episode_reward:.3f} | Success {success} | Steps {step + 1}")

    avg_reward = np.mean([r for r, _, _ in episode_results])
    success_rate = np.mean([1.0 if s else 0.0 for _, s, _ in episode_results])
    print(f"\nCompleted {num_episodes} episodes")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Success rate: {success_rate:.2%}")


def parse_args():
    parser = argparse.ArgumentParser(description='Test PPO actor on Franka Push env')
    parser.add_argument('--actor-path', type=str, required=True, help='Path to saved actor weights (.pth)')
    parser.add_argument('--reward-type', type=str, default='dense', help='FrankaPushEnv reward_type')
    parser.add_argument('--render-mode', type=str, default='human', help='Render mode for the environment')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to evaluate')
    parser.add_argument('--max-steps', type=int, default=50, help='Max steps per episode')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.actor_path):
        raise FileNotFoundError(f"Actor weights not found: {args.actor_path}")

    print('Loading actor from', args.actor_path)
    start = time.time()
    evaluate(
        actor_path=args.actor_path,
        reward_type=args.reward_type,
        render_mode=args.render_mode,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
    )
    print(f"Test finished in {time.time() - start:.1f}s")
