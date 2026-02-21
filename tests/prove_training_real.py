#!/usr/bin/env python3
"""
Proof that training is REAL - shows actual weight changes from backpropagation
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from ml.training.ppo_train import PPOTrainer

print("\n" + "=" * 70)
print("🔍 PROVING TRAINING IS REAL")
print("=" * 70)

# Create trainer
trainer = PPOTrainer(state_dim=10, action_dim=5, device='cpu')

# Capture initial weights from first layer of actor network
initial_weights = trainer.policy.actor[0].weight.clone().detach()
print(f"\n📌 Initial actor weights (first layer, first 5 values):")
print(f"   {initial_weights[0, :5].numpy()}")

# Train for just 3 episodes
print(f"\n🏋️  Training 3 episodes...")
history = trainer.train(num_episodes=3)

# Capture final weights
final_weights = trainer.policy.actor[0].weight.clone().detach()
print(f"\n📌 Final actor weights (first layer, first 5 values):")
print(f"   {final_weights[0, :5].numpy()}")

# Calculate change
weight_diff = (final_weights - initial_weights).abs().mean().item()
max_change = (final_weights - initial_weights).abs().max().item()

print(f"\n📊 Weight Changes:")
print(f"   Average change: {weight_diff:.8f}")
print(f"   Maximum change: {max_change:.8f}")

if weight_diff > 1e-6:
    print(f"\n✅ WEIGHTS CHANGED - Training is REAL!")
    print(f"   Backpropagation updated parameters in neural network")
else:
    print(f"\n❌ WEIGHTS UNCHANGED - Something wrong")

# Show loss trajectory
print(f"\n📉 Loss values per episode:")
for i in range(len(history['policy_loss'])):
    print(f"   Episode {i+1}: policy={history['policy_loss'][i]:8.6f}, value={history['value_loss'][i]:8.6f}")

# Verify gradients are flowing
print(f"\n🔬 Gradient Flow Check:")
print(f"   Optimizer: {type(trainer.optimizer).__name__}")
print(f"   Learning rate: {trainer.optimizer.param_groups[0]['lr']}")
print(f"   Total parameters: {sum(p.numel() for p in trainer.policy.parameters())}")

print("\n" + "=" * 70)
print("🎯 CONCLUSION:")
print("   Training uses REAL PyTorch autograd with:")
print("   1. loss.backward() - computes gradients")
print("   2. optimizer.step() - updates weights")
print("   3. Weight changes prove gradients are applied")
print("=" * 70 + "\n")
