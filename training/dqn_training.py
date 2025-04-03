import os
import gymnasium as gym
from stable_baselines3 import DQN
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


def train_dqn():
    # Create the environment and wrap with Monitor for logging
    env = gym.make("LanguageLearningEnv-v0")
    env = Monitor(env, filename="./training/logs/dqn_monitor_logs")

    # Define the DQN model with tuned hyperparameters
    model = DQN(
        policy="MlpPolicy",  # Multi-layer perceptron policy
        env=env,
        learning_rate=0.0005,  # Learning rate for Q-network
        buffer_size=100000,    # Replay buffer size for 3D environment
        learning_starts=1000,  # Start learning after 1000 steps
        batch_size=64,         # Batch size for training
        tau=1.0,               # Soft update coefficient for target network
        gamma=0.99,            # Discount factor
        exploration_fraction=0.2,  # Fraction of total steps for epsilon decay
        exploration_initial_eps=1.0,  # Initial exploration epsilon
        exploration_final_eps=0.02,   # Final exploration epsilon
        target_update_interval=1000,  # Update target network every 1000 steps
        verbose=1,  # Print training info
        tensorboard_log="./training/tensorboard_logs/dqn/"  # TensorBoard logging
    )

    # Define callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10,000 steps
        save_path="./models/dqn/",
        name_prefix="dqn_language_model",
        save_replay_buffer=True  # Save replay buffer for resuming
    )
    eval_env = gym.make("LanguageLearningEnv-v0")
    eval_env = Monitor(eval_env, filename="./training/logs/dqn_eval_logs")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/dqn/best_model/",
        log_path="./training/logs/dqn_eval_logs",
        eval_freq=5000,  # Evaluate every 5000 steps
        deterministic=True,
        render=False
    )

    # Train the model
    print("Starting DQN training...")
    model.learn(
        total_timesteps=100000,  # Train for 100,000 steps
        callback=[checkpoint_callback, eval_callback],
        log_interval=100  # Log every 100 episodes
    )

    # Save the final model
    final_model_path = "./models/dqn/dqn_language_model"
    model.save(final_model_path)
    print(f"DQN model saved to {final_model_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("./models/dqn/best_model/", exist_ok=True)
    os.makedirs("./training/logs/", exist_ok=True)
    os.makedirs("./training/tensorboard_logs/dqn/", exist_ok=True)

    train_dqn()
