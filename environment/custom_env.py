import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from environment.rendering import LanguageLearningRenderer


class LanguageLearningEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        # 0: Vocabulary, 1: Conversation, 2: Grammar, 3: Culture
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            # [state, x, y, z, performance, engagement, time]
            low=np.array([0, -5, -5, -5, 0, 0, 0]),
            high=np.array([4, 5, 5, 5, 100, 100, 1]),
            dtype=np.float32
        )

        self.vocabulary_by_level = [
            ["Muraho", "Amakuru"], ["Mwaramutse", "Ndagukunda"],
            ["Ndashaka kugura", "Umuryango"], [
                "Ndategereje kuzabona", "Kubera iki"],
            ["Gukoresha neza ururimi", "Gusangira ibitekerezo"]
        ]
        self.grammar_by_level = [
            "Basic greetings", "Simple present tense", "Past tense",
            "Conditionals", "Idiomatic expressions"
        ]
        self.cultural_contexts = [
            "Greetings etiquette", "Family routines", "Traditional celebrations",
            "Historical contexts", "Cultural nuances"
        ]
        self.conversation_topics = [
            "Greetings", "Daily activities", "Personal interests",
            "Current events", "Abstract concepts"
        ]

        self.render_mode = render_mode
        self.renderer = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = 0  # Proficiency level (0-4)
        # 3D position (x, y, z)
        self.position = np.array([-4.0, 0.0, 0.5], dtype=np.float32)
        self.performance = 50.0
        self.engagement = 70.0
        self.time_spent = 0.0  # Normalized 0-1 (90 minutes max)
        self.cumulative_reward = 0
        self.error_count = 0
        self.total_steps = 0
        self.last_action = -1

        if self.render_mode == "human" and self.renderer is None:
            self.renderer = LanguageLearningRenderer(800, 600)

        return self._get_observation(), {}

    def _get_observation(self):
        return np.array([
            self.current_state,
            self.position[0], self.position[1], self.position[2],
            self.performance, self.engagement, self.time_spent / 90.0
        ], dtype=np.float32)

    def step(self, action):
        # Convert action to scalar if it's a NumPy array
        if isinstance(action, np.ndarray):
            action = action.item()

        self.total_steps += 1
        self.time_spent += 1.0
        reward = 0
        terminated = False
        truncated = False
        success = False

        # Define target positions for actions
        action_targets = {
            0: (-2, 2, 0),  # Vocabulary
            1: (2, 2, 0),   # Conversation
            2: (-2, -2, 0),  # Grammar
            3: (2, -2, 0)   # Culture
        }

        # Move towards the target based on action
        target = np.array(action_targets[action], dtype=np.float32)
        direction = target - self.position[:3]
        distance = np.linalg.norm(direction)
        if distance > 0.1:
            self.position[:3] += direction * \
                min(0.1, distance) / distance  # Move 0.1 units max

        # Action effects
        level = int(self.current_state)
        if action == 0:  # Vocabulary
            success_rate = min(
                0.9, 0.5 + (self.engagement / 200) - (level * 0.1))
            success = random.random() < success_rate
            reward += 8 if success else -7
            self.performance = min(
                100, self.performance + (5 if success else -5))
            self.engagement = min(100, self.engagement +
                                  (3 if success else -5))
            self.error_count = 0 if success else self.error_count + 1

        elif action == 1:  # Conversation
            success_rate = min(
                0.85, 0.4 + (self.performance / 200) + (level * 0.05))
            success = random.random() < success_rate
            reward += 12 if success else -7
            self.performance = min(
                100, self.performance + (8 if success else -3))
            self.engagement = min(100, self.engagement +
                                  (5 if success else -2))
            self.error_count = 0 if success else self.error_count + 1

        elif action == 2:  # Grammar
            success_rate = min(0.8, 0.3 + (self.performance / 150))
            success = random.random() < success_rate
            reward += 10 if success else -7
            self.performance = min(
                100, self.performance + (7 if success else -4))
            self.engagement = min(100, self.engagement +
                                  (2 if success else -4))
            self.error_count = 0 if success else self.error_count + 1

        elif action == 3:  # Culture
            success_rate = min(0.9, 0.6 + (self.engagement / 250))
            success = random.random() < success_rate
            reward += 9 if success else -3
            self.performance = min(
                100, self.performance + (4 if success else -2))
            self.engagement = min(100, self.engagement +
                                  (8 if success else -1))
            self.error_count = 0 if success else self.error_count + 1

        # Level progression
        if self.performance >= 80 and self.total_steps % 10 == 0 and self.current_state < 4:
            self.current_state += 1
            reward += 20
            self.performance = max(60, self.performance - 15)
            self.position[0] = -4 + self.current_state * 2  # Move along x-axis

        if self.total_steps % 5 == 0:
            reward += 15
            self.engagement = min(100, self.engagement + 10)

        self.cumulative_reward += reward

        # Termination conditions
        if self.current_state == 4 and self.performance >= 90:
            terminated = True
            reward += 50
        if self.error_count >= 4 or self.time_spent >= 90:
            truncated = True

        info = {
            'success': success, 'current_state': self.current_state,
            'position': self.position.copy(), 'performance': self.performance,
            'engagement': self.engagement, 'cumulative_reward': self.cumulative_reward,
            'last_action': action
        }
        self.last_action = action

        if self.render_mode in ["human", "rgb_array"]:
            self.render()

        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.renderer:
                self.renderer.render_dynamic_scene(
                    current_level=self.current_state,
                    position=self.position,
                    performance=self.performance,
                    engagement=self.engagement,
                    reward=self.cumulative_reward,
                    last_action=self.last_action
                )
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = LanguageLearningRenderer(800, 600)
            self.renderer.render_dynamic_scene(
                current_level=self.current_state,
                position=self.position,
                performance=self.performance,
                engagement=self.engagement,
                reward=self.cumulative_reward,
                last_action=self.last_action
            )
            return self.renderer.save_screenshot("temp.png", return_array=True)

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None
