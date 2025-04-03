# Maxime Bakunzi RL Summative Project

Hey, welcome to my Reinforcement Learning (RL) project! I’m Maxime Bakunzi, and this is my summative assignment where I built a 3D environment to teach an agent Kinyarwanda—my native language—using RL. I compared two methods, Deep Q-Network (DQN) and Proximal Policy Optimization (PPO), to see which one helps the agent learn better. It’s all visualized in 3D with PyOpenGL, and I’ve got videos to prove it works! Check it out below.

## Project Overview

The goal? Create a custom 3D environment where an agent starts as a Kinyarwanda beginner and levels up to fluent by practicing vocabulary, conversation, grammar, and culture. I used DQN (value-based) and PPO (policy gradient) from Stable Baselines3 to train it, then compared their performance. The agent moves in a 3D space, and you can see it live or in saved videos. Spoiler: PPO rocked it!

## Project Structure

Here’s how it’s laid out:

```
maxime_bakunzi_rl_summative/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium env for Kinyarwanda learning
│   ├── rendering.py             # PyOpenGL 3D visualization (static + dynamic)
├── training/
│   ├── dqn_training.py          # DQN training script
│   ├── pg_training.py           # PPO training script
├── models/
│   ├── dqn/                     # Saved DQN models (dqn_language_model.zip)
│   └── pg/                      # Saved PPO models (pg_language_model.zip)
├── video/
│   ├── static_visualization.mp4 # 5-sec static demo of the env
│   ├── dqn_simulation.mp4       # 30-sec DQN sim video
│   └── ppo_simulation.mp4       # 30-sec PPO sim video
├── main.py                      # Runs eval, interactive sims, and saves videos
├── requirements.txt             # Dependencies to install
├── static_visualization_screenshot.jpg # Screenshot from static video (optional)
└── README.md                    # You’re reading it!
```

## Environment Details

- **Agent**: A yellow sphere learning Kinyarwanda, starting at Beginner (-4, 0, 0.5), aiming for Fluent (4, 0, 0.5).
- **Actions**: Discrete (4): Vocabulary (0), Conversation (1), Grammar (2), Culture (3).
- **State**: 7D vector—level (0-4), 3D position (x, y, z), performance (0-100), engagement (0-100), time (0-1).
- **Rewards**: +8 to +12 for success, -3 to -7 for failure, +20 for leveling up, +50 for fluency.
- **Visualization**: 3D with PyOpenGL—proficiency path, action cubes, stats panel.

## How to Run It

### Prerequisites
1. Clone this repo:
   ```bash
   git clone https://github.com/[yourusername]/maxime_bakunzi_rl_summative.git
   cd maxime_bakunzi_rl_summative
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Steps
1. **See the Static Demo**:
   ```bash
   python environment/rendering.py
   ```
   - Pops up a 5-sec 3D view of the env, saves `video/static_visualization.mp4`.

2. **Train the Models** (optional, pre-trained models included):
   - DQN:
     ```bash
     python training/dqn_training.py
     ```
   - PPO:
     ```bash
     python training/pg_training.py
     ```
   - Takes ~100k steps each, saves to `models/dqn/` and `models/pg/`.

3. **Run the Full Demo**:
   ```bash
   python main.py
   ```
   - Evaluates DQN and PPO (10 episodes).
   - Shows interactive 3D sims (close windows to proceed).
   - Saves 30-sec videos to `video/dqn_simulation.mp4` and `video/ppo_simulation.mp4`.

### Example Output (from `main.py`)
```
DQN - Mean Reward: 541.10 ± 91.54
PPO - Mean Reward: 671.20 ± 57.33
Difference (PPO - DQN): 130.10
Video saved as video/dqn_simulation.mp4 with 900 frames (~30.0 seconds)
Video saved as video/ppo_simulation.mp4 with 900 frames (~30.0 seconds)
```

## Static Visualization
Here’s a peek at the static scene (`video/static_visualization.mp4`):

![Static Visualization](static_visualization_screenshot.jpg)  
*The agent at Intermediate, with action cubes and stats.*

## Results
- **DQN**: Solid but shaky (541.10 ± 91.54). Loves exploring but less consistent.
- **PPO**: The winner (671.20 ± 57.33)! Steady and strong—better at mastering Kinyarwanda.

Check my [report PDF](link-to-your-pdf) for the full scoop!

## Dependencies
See `requirements.txt`—key ones are `gymnasium`, `stable-baselines3`, `pygame`, `pyopengl`, and `imageio`.

## Notes
- Videos might warn about resizing (800x600 to 800x608)—it’s fine, just a codec thing.
- Need graphics support for PyOpenGL (update drivers if it’s blank).
- Questions? Hit me up!

Enjoy exploring my Kinyarwanda RL adventure!
```

---

### `requirements.txt`

```plaintext
gymnasium==0.29.1
stable-baselines3==2.3.0
pygame==2.5.2
pyopengl==3.1.7
numpy==1.26.4
imageio==2.34.0
tensorflow==2.16.1  # Optional, used by Stable Baselines3 internally
```

#### Notes on Dependencies
- **`gymnasium`**: For the custom environment.
- **`stable-baselines3`**: RL algorithms (DQN, PPO).
- **`pygame`**: Window management for PyOpenGL.
- **`pyopengl`**: 3D rendering.
- **`numpy`**: Math and array ops.
- **`imageio`**: Video saving.
- **`tensorflow`**: Optional but included since Stable Baselines3 uses it (your `main.py` output shows TensorFlow logs).

Versions are pinned to what’s likely compatible based on your setup (e.g., Python 3.11 from your Conda env). If you’re using a different Python version, test and adjust as needed.

---

### How to Add These Files

1. **Create `README.md`**:
   - Open a text editor (e.g., VS Code).
   - Copy the `README.md` content above.
   - Save as `README.md` in `D:\ALU\ALU Projects\maxime_bakunzi_rl_summative\`.
   - Replace `[yourusername]` with your GitHub username and add your report PDF link if hosted online.

2. **Create `requirements.txt`**:
   - Open a new file in your editor.
   - Copy the `requirements.txt` content above.
   - Save as `requirements.txt` in `D:\ALU\ALU Projects\maxime_bakunzi_rl_summative\`.

3. **Add Screenshot** (Optional):
   - Extract a frame from `video/static_visualization.mp4` (e.g., at 2 seconds) using a video player or tool like VLC (File > Save Snapshot).
   - Name it `static_visualization_screenshot.jpg` and place it in the root folder.
   - Update the `README.md` image path if needed.

4. **Push to GitHub**:
   - If not already a repo:
     ```bash
     git init
     git add .
     git commit -m "Initial commit with full project"
     git remote add origin https://github.com/[yourusername]/maxime_bakunzi_rl_summative.git
     git push -u origin main
     ```
   - If already a repo, just:
     ```bash
     git add README.md requirements.txt static_visualization_screenshot.jpg
     git commit -m "Added README and requirements"
     git push
     ```
