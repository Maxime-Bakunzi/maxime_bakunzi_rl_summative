import os
from environment.custom_env import LanguageLearningEnv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from gymnasium.envs.registration import register
import pygame
import imageio

# Append the project root directory to sys.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "environment"))
sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "training"))


register(
    id="LanguageLearningEnv-v0",
    entry_point="environment.custom_env:LanguageLearningEnv",
)

# Ensure the plots folder exists
os.makedirs("plots", exist_ok=True)

# Paths to your saved models
dqn_model_path = "./models/dqn/dqn_language_model.zip"
ppo_model_path = "./models/pg/pg_language_model.zip"

# Create the environment
env = gym.make("LanguageLearningEnv-v0")

# Load models
print("Loading DQN model...")
dqn_model = DQN.load(dqn_model_path, env=env)
print("Loading PPO model...")
ppo_model = PPO.load(ppo_model_path, env=env)


def run_episodes(model, n_episodes=20):
    """Run evaluation episodes for a given model and record cumulative rewards per episode."""
    rewards = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return rewards


# Run evaluation episodes for both models
n_eval_episodes = 20
print("Running evaluation episodes for DQN...")
dqn_rewards = run_episodes(dqn_model, n_eval_episodes)
print("Running evaluation episodes for PPO...")
ppo_rewards = run_episodes(ppo_model, n_eval_episodes)

# Plot: Cumulative Rewards per Episode
episodes = np.arange(1, n_eval_episodes + 1)
plt.figure(figsize=(8, 5))
plt.plot(episodes, dqn_rewards, label='DQN',
         color='blue', marker='o', markersize=5)
plt.plot(episodes, ppo_rewards, label='PPO',
         color='green', marker='o', markersize=5)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward per Episode')
plt.legend()
plt.grid(True)
plt.savefig("plots/cumulative_rewards.png")
plt.close()
print("Saved cumulative rewards plot to plots/cumulative_rewards.png")
