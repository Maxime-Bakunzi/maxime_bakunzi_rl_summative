# Kinyarwanda Language Learning RL Project

**Video Recording (3 min):** [Link to your video](#)

---

## Overview

Hey there! Welcome to my Kinyarwanda Language Learning RL project. This project is very close to my heart since it combines my passion for my native language, Kinyarwanda, with my interest in artificial intelligence and education. I built a 3D environment where an agent learns Kinyarwanda by practicing vocabulary, conversation, grammar, and cultural aspects—much like a virtual tutor. 

I experimented with two reinforcement learning methods:
- **Deep Q-Network (DQN):** A value-based approach.
- **Proximal Policy Optimization (PPO):** A policy gradient method.

The main idea is to see which approach helps the agent master Kinyarwanda faster and more effectively. This project is also a stepping stone towards my larger mission: creating an assistant web/app that helps African students interact—imagine one student speaking in Kiswahili and another in Kinyarwanda, breaking language barriers and building cross-cultural connections!

---

## Project Structure

```
maxime_bakunzi_rl_summative/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment for language learning
│   ├── rendering.py             # PyOpenGL 3D visualization (static & dynamic)
├── training/
│   ├── dqn_training.py          # DQN training script
│   ├── pg_training.py           # PPO training script
├── models/
│   ├── dqn/                     # Saved DQN models (e.g., dqn_language_model.zip)
│   └── pg/                      # Saved PPO models (e.g., pg_language_model.zip)
├── video/
│   ├── static_visualization.mp4 # 5-second static demo of the environment
│   ├── dqn_simulation.mp4       # 30-second DQN simulation video
│   └── ppo_simulation.mp4       # 30-second PPO simulation video
├── main.py                      # Entry point for evaluation, simulation, and video saving
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation (this file)
```

---

## Environment Description

- **Agent:**  
  The agent represents a student learning Kinyarwanda, visualized as a yellow sphere navigating a 3D space. It starts at a "Beginner" level and aims to reach "Fluent" by interacting with various learning stations.

- **Action Space:**  
  The agent has four discrete actions:  
  0. **Vocabulary:** Learn words (e.g., "Muraho" or "Ndagukunda").  
  1. **Conversation:** Engage in chats about daily life.  
  2. **Grammar:** Practice sentence structures and tenses.  
  3. **Culture:** Explore traditions and etiquette.

- **State Space:**  
  The state is a 7-dimensional vector representing:  
  - Proficiency level (0–4)  
  - 3D coordinates (x, y, z)  
  - Performance (0–100)  
  - Engagement (0–100)  
  - Time spent (normalized)

- **Reward Structure:**  
  Rewards and penalties are assigned based on action success or failure. Special rewards are given for level-ups and engagement boosts, while too many errors or reaching the time limit ends the session.

- **Visualization:**  
  I used PyOpenGL to create an engaging 3D visualization of the environment. The scene includes a proficiency path, action stations (colored cubes), an agent (yellow sphere), and a stats panel. Check out the [static video](video/static_visualization.mp4) for a quick look!

---

## Implemented Methods

### Deep Q-Network (DQN)
- **Approach:**  
  I used Stable Baselines3’s DQN with an MLP policy. The model uses a target network for stability and a replay buffer (100,000 slots) to learn from past experiences.
- **Exploration:**  
  The exploration rate decays from 1.0 to 0.02 over 20% of training steps.
- **State Space:**  
  The model handles the custom 7D state space.

### Proximal Policy Optimization (PPO)
- **Approach:**  
  For policy gradient learning, I used PPO (also via Stable Baselines3) with an MLP policy that learns a probability distribution over actions.
- **Updates:**  
  The model updates every 2048 steps with 10 epochs and includes a small entropy bonus (0.01) to encourage exploration.
- **Performance:**  
  PPO showed a higher and more consistent performance compared to DQN.

---

## Hyperparameter Optimization

### DQN Hyperparameters
| Hyperparameter          | Optimal Value               | Summary                                                                                             |
|-------------------------|-----------------------------|-----------------------------------------------------------------------------------------------------|
| Learning Rate           | 0.0005                      | Stable convergence within 100k steps; higher rates made the training jumpy.                        |
| Gamma (Discount Factor) | 0.99                        | Emphasizes long-term rewards, crucial for level-ups.                                               |
| Replay Buffer Size      | 100,000                     | Sufficient for a 3D environment; smaller sizes lost key experiences.                               |
| Batch Size              | 64                          | Provided a good balance between speed and stability.                                               |
| Exploration Strategy    | ε: 1.0 → 0.02 (20% decay)   | Ensured ample exploration at the start and gradual exploitation later.                             |

### PPO Hyperparameters
| Hyperparameter             | Optimal Value | Summary                                                                                                    |
|----------------------------|---------------|------------------------------------------------------------------------------------------------------------|
| Learning Rate              | 0.0003        | The default value worked great; higher rates led to unstable training.                                   |
| Gamma (Discount Factor)    | 0.99          | Focused on long-term rewards to help the agent reach fluency.                                             |
| N Steps                    | 2048          | A good rollout size for stable updates.                                                                    |
| Batch Size                 | 64            | Maintained fast and stable training.                                                                       |
| Entropy Coefficient        | 0.01          | Encouraged exploration without making the policy too unpredictable.                                        |

---

## Metrics Analysis

- **Cumulative Reward:**  
  - **DQN:** 541.10 ± 91.54  
  - **PPO:** 671.20 ± 57.33  
  PPO outperformed DQN with higher average rewards and lower variance.

- **Training Stability:**  
  PPO’s training was smoother and more consistent, whereas DQN had higher variance indicating less stability.

- **Episodes to Convergence:**  
  Although both methods were trained for 100k steps, PPO reached higher rewards faster, suggesting faster convergence.

- **Generalization:**  
  PPO showed better adaptability when tested on unseen states, while DQN struggled more with new positions in the 3D environment.

---

## How to Run the Project

### Prerequisites
- Python 3.7+
- [Pip](https://pip.pypa.io/en/stable/)
- Dependencies listed in `requirements.txt`

### Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Maxime-Bakunzi/maxime_bakunzi_rl_summative.git
   cd maxime_bakunzi_rl_summative
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Training the Models
- **Train DQN Model:**
  ```bash
  python training/dqn_training.py
  ```
- **Train PPO Model:**
  ```bash
  python training/pg_training.py
  ```

### Running Evaluations and Simulations
Run the main script to evaluate the models, simulate agent interactions, and generate simulation videos:
```bash
python main.py
```

---

## Videos & Visualizations

- **Static Environment Visualization:** [static_visualization.mp4](video/static_visualization.mp4)  
- **DQN Simulation Video:** [dqn_simulation.mp4](video/dqn_simulation.mp4)  
- **PPO Simulation Video:** [ppo_simulation.mp4](video/ppo_simulation.mp4)

Also, check out my **3-minute video** where I explain the project and share insights on the development process: [3-Minute Project Video](#).

---

## Future Vision

This project is part of my broader mission to build an assistant app/web platform that can help African students communicate effectively. Imagine a scenario where one student speaks Kiswahili while another responds in Kinyarwanda—fostering rich, cross-cultural exchanges. By starting with a focused problem (teaching Kinyarwanda), I hope to lay the groundwork for this larger vision.

---

