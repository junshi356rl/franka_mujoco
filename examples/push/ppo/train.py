import collections
import glob
import os
import sys
import time
from typing import List
import pandas as pd
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../../")))
from panda_mujoco_gym.envs.push import FrankaPushEnv

from utils import (
    MLP, 
    ObsNormalizer, 
    get_action_sample, 
    get_action_prob, 
    action_forward,
    plot_batch_success_rate,
)

def compute_gae_advantage(reward_his: List[float], 
                          state_his: List[torch.Tensor], 
                          gae_lambda: float, 
                          gamma: float, 
                          critic: MLP, 
                          is_truncated: bool, 
                          dev: torch.device):
    # Assume the rollout has T steps
    # reward_his: reward from 0 to T
    # state_his_extended: reward from 0 to T + 1 (included the state after the last step)
    gae_delta_his = []
    gae_advantage_his = []
    with torch.no_grad():
        states_tensor = torch.stack(state_his, dim=0).to(dev)
        old_value_his = critic(states_tensor).squeeze().cpu().numpy()
    for i in range(len(reward_his)):
        if i < (len(reward_his) - 1):
            old_value_next = old_value_his[i+1]
        elif is_truncated:
            old_value_next = old_value_his[i+1]
        else:
            old_value_next = 0
        delta = reward_his[i] \
            + gamma * old_value_next \
            - old_value_his[i]
        gae_delta_his.append(delta)
    gae = 0
    for delta in reversed(gae_delta_his):
        gae = gae * gamma * gae_lambda + delta
        gae_advantage_his.append(gae)
    gae_advantage_his.reverse()
    return gae_advantage_his, old_value_his[:-1]

def get_custom_reward(obs, env_reward):
    ee_position = obs[:3]
    object_position = obs[6:9]
    dist = np.linalg.norm(ee_position - object_position)
    custom_reward = -dist + env_reward  # Combine with env reward to encourage both reaching and pushing
    return custom_reward

def rollout(env: FrankaPushEnv, actor: MLP, critic: MLP, batch_steps, rollout_max_step, 
            action_size, gamma, dev, obs_normalizer: ObsNormalizer, gae_lambda:int):
    batch_state_his = []
    batch_action_his = []
    batch_log_prob_his = []
    batch_gae_advantage_his = []
    batch_old_value_his = []
    batch_reward_his = []
    step = 0
    rollout_count = 0
    success_rollout_count = 0
    while step < batch_steps:
        rollout_count += 1
        state_his = []
        action_his = []
        reward_his = []
        log_prob_his = []
        cur_state, _ = env.reset()
        rollout_step = 0
        cur_obs_np = np.concatenate([cur_state['observation'], cur_state['desired_goal']])
        cur_obs = obs_normalizer.normalize(torch.tensor(cur_obs_np, device=dev, dtype=torch.float32))
        while rollout_step < rollout_max_step:
            actor_output = action_forward(actor, cur_obs)
            actor_distribution_mean = actor_output[:action_size]
            actor_distribution_log_std = actor_output[action_size:]
            action, log_prob = get_action_sample(actor_distribution_mean, actor_distribution_log_std)
            state_his.append(cur_obs)
            xyz_action =  np.append(action.cpu().numpy(), 0.03)  # For push task, we only control x and y, keep z action as 0.03
            cur_state, env_reward, terminated, truncated, _ = env.step(xyz_action)
            cur_obs_np = np.concatenate([cur_state['observation'], cur_state['desired_goal']])
            reward = get_custom_reward(cur_obs_np, env_reward)
            obs_normalizer.update(cur_obs_np[np.newaxis, :])
            cur_obs = obs_normalizer.normalize(torch.tensor(cur_obs_np, device=dev, dtype=torch.float32))
            action_his.append(action)
            reward_his.append(reward)
            log_prob_his.append(log_prob)

            step += 1
            rollout_step += 1

            if terminated:
                success_rollout_count += 1
                break
            elif truncated:
                break
        

        # Pass the next state of the last step to critic to calculate V(s_{t+1}) for GAE advantage calculation
        gae_advantage_his, old_value_his = compute_gae_advantage(
            reward_his, state_his+[cur_obs], gae_lambda, gamma, critic, truncated, dev)
        batch_gae_advantage_his.extend(gae_advantage_his)
        batch_old_value_his.extend(old_value_his)

        batch_state_his.extend(state_his)
        batch_action_his.extend(action_his)
        batch_log_prob_his.extend(log_prob_his)
        batch_reward_his.extend(reward_his)

    success_rate = success_rollout_count / rollout_count if rollout_count > 0 else 0.0
    return (torch.stack(batch_state_his, dim=0),
            torch.stack(batch_action_his, dim=0),
            (torch.tensor(batch_gae_advantage_his, device=dev, dtype=torch.float32), torch.tensor(batch_old_value_his, device=dev, dtype=torch.float32)),
            torch.stack(batch_log_prob_his, dim=0),
            success_rate,
            torch.tensor(batch_reward_his, device=dev, dtype=torch.float32))


if __name__ == "__main__":
    # Init the actor network and critic network
    TOTAL_LOG_NAME = 'log.txt'
    # remove old log
    if os.path.exists(TOTAL_LOG_NAME):
        os.remove(TOTAL_LOG_NAME)
    max_step = 50
    reward_type = "dense"
    state_size = 18 + 3  # obs + goal
    action_size = 2
    gamma = 0.99
    gae_lambda = 0.95
    batch_steps = 2000
    train_total_steps = 2500000
    n_updates = 3
    eps = 0.2
    actor_lr = 3e-4
    critic_lr = 1e-3
    entropy_coef = 0.01
    env = FrankaPushEnv(reward_type=reward_type)
    obs_normalizer = ObsNormalizer(dim=state_size)
    actor = MLP(state_size, action_size, [128, 64], action_size * 2).to(torch.device("cuda")) # Actor output is mean and std of the action distribution
    critic = MLP(state_size, action_size, [128, 64], 1).to(torch.device("cuda"))
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    total_batches = train_total_steps // batch_steps
    actor_scheduler = torch.optim.lr_scheduler.LinearLR(
        actor_optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_batches
    )
    critic_scheduler = torch.optim.lr_scheduler.LinearLR(
        critic_optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_batches
    )
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"ppo_push_output/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    actor_loss_his = []
    critic_loss_his = []
    batch_success_rate_his = []
    std_history = collections.deque(maxlen=20)
    # Main loop of PPO

    # Entropy for diagonal Gaussian: 0.5 * sum(1 + log(2π) + 2*log_std), extract the constant part for later calculation to save time
    GAUSSIAN_ENTROPY_OFFSET = 0.5 * (1.0 + np.log(2 * np.pi))

    total_steps = 0
    sample_batch_count = 0
    while total_steps < train_total_steps:
        # GAE Advantage
        batch_state, batch_action, (A_RAW, V_old), batch_log_prob, batch_success_rate, batch_rewards = rollout(
            env, actor, critic, batch_steps, max_step, action_size, gamma, 
            dev=torch.device("cuda"), obs_normalizer=obs_normalizer, gae_lambda=gae_lambda)
        batch_success_rate_his.append(batch_success_rate)
        reward_std = batch_rewards.std().item()
        
        with torch.no_grad():
            A = (A_RAW - A_RAW.mean()) / (A_RAW.std() + 1e-8)
            critic_target = V_old + A_RAW
        
        kl_list, clip_list = [], []
        for i_updates in range(n_updates):
            old_log_prob = batch_log_prob.detach()
            actor_output = action_forward(actor,batch_state)
            mean = actor_output[:, :action_size]
            log_std = actor_output[:, action_size:]

            new_log_prob = get_action_prob(mean, log_std, batch_action)
            rt = torch.exp(new_log_prob - old_log_prob)

            kl_approx = torch.mean(old_log_prob - new_log_prob).item()
            clip_frac = ((rt < 1 - eps) | (rt > 1 + eps)).float().mean().item()
            kl_list.append(kl_approx)
            clip_list.append(clip_frac)
            
            # Calculate the Clip loss
            clip_loss = (-torch.min(
                rt * A,
                torch.clamp(rt, 1 - eps, 1 + eps) * A
            )).mean()
            entropy = (GAUSSIAN_ENTROPY_OFFSET + log_std).sum(dim=-1)
            # Calculate the Critic loss MSE(V, rtgs)
            V = critic.forward(batch_state).squeeze()
            # GAE Advantage
            critic_loss = torch.nn.functional.mse_loss(V, critic_target)
            actor_loss = clip_loss - entropy_coef * entropy.mean()
            # Gradient descent Clip loss and Critic loss
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=3.0)
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=3.0)
            critic_optimizer.step()
            # Logging
            actor_loss_his.append(actor_loss.mean().item())
            critic_loss_his.append(critic_loss.mean().item())
        
        total_steps += len(batch_state)
        actor_scheduler.step()
        critic_scheduler.step()
        sample_batch_count += 1

        with torch.no_grad():
            V_final = critic(batch_state).squeeze()
            var_target = torch.var(critic_target, unbiased=False)
            var_resid = torch.var(critic_target - V_final, unbiased=False)
            exp_var = 1.0 - var_resid / (var_target + 1e-8)
            std_history.append(A_RAW.std().item())
            std_cv = np.std(std_history) / (np.mean(std_history) + 1e-8) if len(std_history)>=10 else 0.0
            A_RAW_std = A_RAW.std().item()
            A_RAW_q05 = torch.quantile(A_RAW, 0.05).item()
            A_RAW_q95 = torch.quantile(A_RAW, 0.95).item()
            A_RAW_range = A_RAW_q95 - A_RAW_q05
            ratio_a_r = A_RAW_std / (reward_std + 1e-8)
            grad_scale = sum(p.grad.abs().mean().item() for p in actor.parameters() if p.grad is not None)

        if sample_batch_count % 5 == 0:
            print(f"\n{'='*70}")
            print(f"📦 Batch {sample_batch_count:03d} | Steps: {total_steps} | Success: {batch_success_rate:.1%}")
            print(f"🧠 Policy  -> Loss: {actor_loss.item():.4f} | Entropy: {entropy.mean().item():.3f} | Avg KL: {np.mean(kl_list):.4f} | ClipFrac: {np.mean(clip_list):.1%}")
            print(f"💰 Critic  -> Loss: {critic_loss.item():.4f} | ExplainedVar: {exp_var.item():.3f} | TargetAbsMean: {critic_target.abs().mean().item():.2f}")
            print(f"⚖️ GAE     -> A_std: {A_RAW_std:.3f} | r_std: {reward_std:.3f} | Ratio(A/r): {ratio_a_r:.2f}")
            print(f"📉 Grad    -> Actor: {actor_grad_norm:.3f} | Critic: {critic_grad_norm:.3f}")
            print(f"[Grad Scale] Actor raw grad: {grad_scale:.4f} | After clip: {actor_grad_norm:.4f}")
            print(f"{'='*70}")

            # Create DataFrame for logging batch data
            data = {
                'batch_state': batch_state.cpu().numpy().tolist(),
                'batch_action': batch_action.cpu().numpy().tolist(),
                'batch_log_prob': batch_log_prob.cpu().numpy().tolist(),
                'batch_success_rate': [batch_success_rate] * len(batch_state),
                'A': A.cpu().numpy().tolist(),
                'old_log_prob': old_log_prob.cpu().numpy().tolist(),
                'actor_output': actor_output.detach().cpu().numpy().tolist(),
                'new_log_prob': new_log_prob.detach().cpu().numpy().tolist(),
                'rt': rt.detach().cpu().numpy().tolist(),
                'clip_loss': [clip_loss.detach().item()] * len(batch_state),
                'entropy': entropy.detach().cpu().numpy().tolist(),
                'V': V.detach().cpu().numpy().tolist(),
                'critic_loss': [critic_loss.detach().item()] * len(batch_state),
                'actor_loss': [actor_loss.detach().item()] * len(batch_state),
                'actor_grad_norm': [actor_grad_norm] * len(batch_state),
                'critic_grad_norm': [critic_grad_norm] * len(batch_state),
            }
            df = pd.DataFrame(data)
            df.to_csv(f"{output_dir}/batch_log_{sample_batch_count}.csv", index=False)

    # Save batch success rate curve
    plot_batch_success_rate(batch_success_rate_his, output_path=f"{output_dir}/batch_success_rate.png")

    # Save network weights
    torch.save(actor.state_dict(), f"{output_dir}/ppo_push_actor.pth")
    torch.save(critic.state_dict(), f"{output_dir}/ppo_push_critic.pth")

    # Save ObsNormalizer state
    obs_normalizer.save(f"{output_dir}/ppo_push_normalizer.pkl")

    # Merge all batch log CSVs into a total CSV
    all_files = glob.glob(f"{output_dir}/batch_log_*.csv")
    if all_files:
        df_list = [pd.read_csv(file) for file in all_files]
        total_df = pd.concat(df_list, ignore_index=True)
        total_df.to_csv(f"{output_dir}/total_batch_log.csv", index=False)