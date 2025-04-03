import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.envs.registration import register
import sys
# Append the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# Register the custom environment
register(
    id="LanguageLearningEnv-v0",
    entry_point="environment.custom_env:LanguageLearningEnv",
)


def train_ppo():
    # Create the environment and wrap with Monitor for logging
    env = gym.make("LanguageLearningEnv-v0")
    env = Monitor(env, filename="./training/logs/ppo_monitor_logs")

    # Define the PPO model with tuned hyperparameters
    model = PPO(
        policy="MlpPolicy",  # Multi-layer perceptron policy
        env=env,
        learning_rate=0.0003,  # Learning rate for policy and value networks
        n_steps=2048,          # Steps per update (rollout buffer size)
        batch_size=64,         # Mini-batch size for optimization
        n_epochs=10,           # Number of epochs per update
        gamma=0.99,            # Discount factor
        gae_lambda=0.95,       # Generalized Advantage Estimation lambda
        ent_coef=0.01,         # Entropy coefficient for exploration
        verbose=1,             # Print training info
        tensorboard_log="./training/tensorboard_logs/ppo/"  # TensorBoard logging
    )

    # Define callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10,000 steps
        save_path="./models/pg/",
        name_prefix="ppo_language_model"
    )
    eval_env = gym.make("LanguageLearningEnv-v0")
    eval_env = Monitor(eval_env, filename="./training/logs/ppo_eval_logs")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/pg/best_model/",
        log_path="./training/logs/ppo_eval_logs",
        eval_freq=5000,  # Evaluate every 5000 steps
        deterministic=True,
        render=False
    )

    # Train the model
    print("Starting PPO training...")
    model.learn(
        total_timesteps=100000,  # Train for 100,000 steps
        callback=[checkpoint_callback, eval_callback],
        log_interval=100  # Log every 100 episodes
    )

    # Save the final model
    final_model_path = "./models/pg/pg_language_model"
    model.save(final_model_path)
    print(f"PPO model saved to {final_model_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("./models/pg/best_model/", exist_ok=True)
    os.makedirs("./training/logs/", exist_ok=True)
    os.makedirs("./training/tensorboard_logs/ppo/", exist_ok=True)

    train_ppo()
