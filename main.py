from environment.custom_env import LanguageLearningEnv
import os
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
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


def evaluate_and_compare_models(dqn_path, ppo_path, n_eval_episodes=10):
    env = gym.make("LanguageLearningEnv-v0")
    env = Monitor(env)

    print("Loading DQN model...")
    dqn_model = DQN.load(dqn_path, env=env)
    print("Loading PPO model...")
    ppo_model = PPO.load(ppo_path, env=env)

    print(f"Evaluating DQN over {n_eval_episodes} episodes...")
    dqn_mean_reward, dqn_std_reward = evaluate_policy(
        dqn_model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"DQN - Mean Reward: {dqn_mean_reward:.2f} ± {dqn_std_reward:.2f}")

    print(f"Evaluating PPO over {n_eval_episodes} episodes...")
    ppo_mean_reward, ppo_std_reward = evaluate_policy(
        ppo_model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"PPO - Mean Reward: {ppo_mean_reward:.2f} ± {ppo_std_reward:.2f}")

    print("\nPerformance Comparison:")
    print(f"DQN: {dqn_mean_reward:.2f} ± {dqn_std_reward:.2f}")
    print(f"PPO: {ppo_mean_reward:.2f} ± {ppo_std_reward:.2f}")
    print(f"Difference (PPO - DQN): {(ppo_mean_reward - dqn_mean_reward):.2f}")

    env.close()
    return {"DQN": (dqn_mean_reward, dqn_std_reward), "PPO": (ppo_mean_reward, ppo_std_reward)}


def simulate_agent(model, render_mode="human", output_video=None, target_frames=900):
    env = gym.make("LanguageLearningEnv-v0", render_mode=render_mode)
    env = Monitor(env)
    frames = []
    total_steps = 0

    obs, _ = env.reset()
    while total_steps < target_frames:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1

        if render_mode == "rgb_array":
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        if terminated or truncated:
            obs, _ = env.reset()

        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

    if output_video and frames:
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        writer = imageio.get_writer(output_video, fps=30)
        for frame in frames[:target_frames]:
            writer.append_data(frame)
        writer.close()
        print(
            f"Video saved as {output_video} with {len(frames)} frames (~{len(frames)/30:.1f} seconds)")

    env.close()


def main():
    dqn_model_path = "./models/dqn/dqn_language_model.zip"
    ppo_model_path = "./models/pg/pg_language_model.zip"

    if not os.path.exists(dqn_model_path) or not os.path.exists(ppo_model_path):
        print("Error: One or both model files not found. Please check paths.")
        return

    results = evaluate_and_compare_models(
        dqn_model_path, ppo_model_path, n_eval_episodes=10)

    env = gym.make("LanguageLearningEnv-v0")
    env = Monitor(env)
    dqn_model = DQN.load(dqn_model_path, env=env)
    ppo_model = PPO.load(ppo_model_path, env=env)

    print("\nSimulating DQN agent interactively (close window to proceed)...")
    simulate_agent(dqn_model, render_mode="human")
    print("Saving DQN simulation video...")
    simulate_agent(dqn_model, render_mode="rgb_array",
                   output_video="video/dqn_simulation.mp4")

    print("\nSimulating PPO agent interactively (close window to proceed)...")
    simulate_agent(ppo_model, render_mode="human")
    print("Saving PPO simulation video...")
    simulate_agent(ppo_model, render_mode="rgb_array",
                   output_video="video/ppo_simulation.mp4")

    print("\nExperiment Summary:")
    print(
        f"DQN Results: Mean Reward = {results['DQN'][0]:.2f} ± {results['DQN'][1]:.2f}")
    print(
        f"PPO Results: Mean Reward = {results['PPO'][0]:.2f} ± {results['PPO'][1]:.2f}")
    print("Simulations completed and videos saved in 'video/' folder.")


if __name__ == "__main__":
    main()
