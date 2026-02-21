"""
Offline PPO Training for StreamSafe-RL
Train PPO policy on logged moderation decisions using PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """
    PPO Actor-Critic Network for discrete action space
    Action space: [IGNORE, WARN, TIMEOUT_60S, TIMEOUT_600S, BAN]
    """

    def __init__(self, state_dim: int = 10, hidden_dim: int = 64, action_dim: int = 5):
        super().__init__()

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returns both action probabilities and state value
        """
        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value


class PPOTrainer:
    """
    Offline PPO Trainer using logged decision data
    """
    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 5,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        device: str = 'cpu'
    ):
        self.device = torch.device(device)
        self.policy = PolicyNetwork(state_dim, hidden_dim=64, action_dim=action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        logger.info(f"PPO Trainer initialized on {self.device}")

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """
        Compute discounted returns (Monte Carlo estimation)
        """
        returns = []
        R = 0
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        # Normalize returns for stable training
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def train_on_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """
        PPO update step using clipped surrogate objective
        """
        metrics = {'policy_loss': 0.0, 'value_loss': 0.0}

        for _ in range(self.k_epochs):
            # Get current policy predictions
            action_probs, state_values = self.policy(states)
            state_values = state_values.squeeze()

            # Get log probabilities of actions taken
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)

            # PPO clipped surrogate objective
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            advantages = returns - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(state_values, returns)

            loss = policy_loss + 0.5 * value_loss

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

            metrics['policy_loss'] += policy_loss.item()
            metrics['value_loss'] += value_loss.item()
        
        # Average metrics over epochs
        for key in metrics:
            metrics[key] /= self.k_epochs

        return metrics
    
    def save_checkpoint(self, path: str):
        """
        Save model checkpoint
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def train(self, num_episodes: int = 100, batch_size: int = 256) -> Dict[str, List[float]]:
        """
        Main training loop using synthetic data
        (In production, load from parquet logs)
        """
        logger.info(f"Starting offline PPO training for {num_episodes} episodes")

        # Generate synthetic data for demonstration
        num_samples = 10000
        states = np.random.rand(num_samples, 10).astype(np.float32)
        actions = np.random.randint(0, 5, size=num_samples)
        rewards = np.where(np.random.rand(num_samples) < 0.2, 0.1, -1.0).astype(np.float32)

        history = {'episode_reward': [], 'policy_loss': [], 'value_loss': []}

        for episode in range(num_episodes):
            # Sample batch
            indices = np.random.choice(len(states), batch_size, replace=False)
            batch_states = torch.tensor(states[indices]).to(self.device)
            batch_actions = torch.tensor(actions[indices]).to(self.device)
            batch_rewards = rewards[indices].tolist()

            # Compute old log probabilities
            with torch.no_grad():
                actions_probs, _ = self.policy(batch_states)
                dist = Categorical(actions_probs)
                old_log_probs = dist.log_prob(batch_actions)

            # Compute returns
            returns = self.compute_returns(batch_rewards)

            # PPO update
            metrics = self.train_on_batch(batch_states, batch_actions, old_log_probs, returns)

            # Log metrics
            history['episode_reward'].append(np.mean(batch_rewards))
            history['policy_loss'].append(metrics['policy_loss'])
            history['value_loss'].append(metrics['value_loss'])

            if (episode + 1) % 10 == 0:
                logger.info(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Reward: {history['episode_reward'][-1]:.3f} | "
                    f"Loss: {metrics['policy_loss']:.4f}"
                )
        
        logger.info("Training completed")
        return history


def main():
    """
    Main training script
    """
    import argparse

    parser = argparse.ArgumentParser(description="Offline PPO Training for StreamSafe-RL")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--output", type=str, default="models/ppo_policy.pt", help="Output model path")
    args = parser.parse_args()

    # Initialize trainer
    trainer = PPOTrainer(
        state_dim=10,
        action_dim=5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Train
    history = trainer.train(num_episodes=args.episodes)

    # Save checkpoint
    trainer.save_checkpoint(args.output)

    logger.info(f"Training complete. Model saved to {args.output}")


if __name__ == "__main__":
    main()
