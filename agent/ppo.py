import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from .actor_critic import ActorCritic

EPS = 1e-6

class PPO:
    """
    Pseudocode:
    1. Collect trajectories
    2. Compute advantages using GAE
    3. Update policy using clipped objective
    4. Update value function
    5. Train the model 
    """
    
    def __init__(self, env, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                 n_epochs=10, batch_size=64, ent_coef=0.01, vf_coef=0.5):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.policy = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
    
    def compute_gae(self, rewards, values, dones):
        """
        GAE Formula: A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD error
        """

        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))): 
        # for t in range(len(rewards) - 1, -1, -1):
            if dones[t]: 
                next_value = 0  # Terminal state has no future value 
            else: 
                next_value = values[t + 1] 
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)  # Insert at beginning since we're going backwards 
        
        advantages = np.array(advantages, dtype=np.float32)
        values_arr = np.array(values[:-1], dtype=np.float32)
        returns = advantages + values_arr  # R_t = A_t + V(s_t)

        # Normalize advantages 
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns 
    
    def collect_trajectories(self, n_steps):
        # 1: Initialize storage lists
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        # 2: Reset environment and get initial state
        state, _ = self.env.reset() 
        
        # 3: Collect n_steps of experience
        for step in range(n_steps): 
            # 4: Convert state to tensor and get action from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0) 
            with torch.no_grad(): 
                action, log_prob, value = self.policy.get_action_and_log_prob_and_value(state_tensor) 
                action = action.cpu().numpy()[0]
                log_prob = log_prob.cpu().item()
                value = value.cpu().numpy()[0, 0]
            
            # 5: Take action in environment
            next_state, reward, done, truncated, _ = self.env.step(action) 
            # 6: Store transition
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done or truncated)

            # 7: Update state (reset if episode ended)
            state = next_state
            if done or truncated:
                state, _ = self.env.reset()
            
        # Get final value estimate 
        with torch.no_grad():
            final_state_tensor = torch.FloatTensor(state).unsqueeze(0)
            _, _, final_value = self.policy.get_action_and_log_prob_and_value(final_state_tensor)
            values.append(final_value.cpu().numpy()[0, 0])
            
        return {
            'states': np.array(states, dtype=np.float32),
            'actions': np.array(actions, dtype=np.float32),
            'log_probs': np.array(log_probs, dtype=np.float32),
            'rewards': rewards,
            'values': values,
            'dones': dones
        }


    def update_policy(self, states, actions, old_log_probs, advantages, returns):
        """  
        PPO Objective: L^{CLIP}(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
        where r(θ) = π_θ(a|s) / π_θ_old(a|s) is the probability ratio
        """
        # Convert data to tensors 
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        dataset = TensorDataset(states, actions, old_log_probs, advantages, returns)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        policy_losses, value_losses, entropy_losses = [], [], []
        for epoch in range(self.n_epochs):
            for batch in dataloader:
                batch_states, batch_actions, batch_old_lp, batch_adv, batch_ret = batch

                # 1: Get current distribution and value from policy
                dist, batch_values = self.policy.forward(batch_states)
                
                # inverse-tanh actions to regain log-prob
                a_clamped = torch.clamp(batch_actions, -1 + EPS, 1 - EPS)
                pre_tanh = torch.atanh(a_clamped)
                batch_log_probs = (dist.log_prob(pre_tanh) - torch.log(1 - a_clamped.pow(2) + EPS)).sum(dim=-1)

                # Entropy bonus (encourages exploration)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # 3: Compute probability ratio r(θ) = π_new / π_old
                ratio = torch.exp(batch_log_probs - batch_old_lp)

                # 4: Compute clipped surrogate objective
                surr1 = ratio * batch_adv 
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 5: Compute value function loss (MSE)
                value_loss = 0.5 * ((batch_values.squeeze(-1) - batch_ret) ** 2).mean()

                # 6: Total loss with entropy bonus
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                # 7: Backpropagate and optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses)
        }
    
    def train(self, n_iterations, steps_per_iter=2048):
        # 1: Initialize tracking lists
        training_stats = {
            'iterations': [],
            'mean_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': []
        }
        
        # 2: Main training loop
        for iteration in range(n_iterations): 
            # 3: Collect trajectories
            trajectories = self.collect_trajectories(steps_per_iter)

            # 4: Compute advantages
            advantages, returns = self.compute_gae(
                trajectories['rewards'],
                trajectories['values'],
                trajectories['dones']
            )

            # 5: Update policy 
            update_stats = self.update_policy(
                trajectories['states'],
                trajectories['actions'],
                trajectories['log_probs'],
                advantages,
                returns
            )

            # 6: Log training statistics
            mean_reward = np.mean(trajectories['rewards'])
            training_stats['iterations'].append(iteration)
            training_stats['mean_rewards'].append(mean_reward)
            training_stats['policy_losses'].append(update_stats['policy_loss'])
            training_stats['value_losses'].append(update_stats['value_loss'])
            training_stats['entropies'].append(update_stats['entropy'])

            # 7: Print training progress
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}/{n_iterations} | "
                      f"Mean Reward: {mean_reward:.2f} | "
                      f"Policy Loss: {update_stats['policy_loss']:.4f} | "
                      f"Value Loss: {update_stats['value_loss']:.4f} | "
                      f"Entropy: {update_stats['entropy']:.4f}")

        return training_stats

    def predict(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist, _ = self.policy.forward(state_tensor)
            # Get mean of the distribution, no need to sample
            action = torch.tanh(dist.mean)  
        return action.cpu().numpy()[0]
