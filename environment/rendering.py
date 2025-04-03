import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import os
import imageio


class LanguageLearningRenderer:
    def __init__(self, window_width=800, window_height=600):
        pygame.init()
        self.window_width = window_width
        self.window_height = window_height
        self.display = (window_width, window_height)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption(
            "Kinyarwanda Language Learning - 3D Visualization")

        glViewport(0, 0, window_width, window_height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (window_width / window_height), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, -8, 4, 0, 0, 0, 0, 0, 1)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_DEPTH_TEST)
        glLightfv(GL_LIGHT0, GL_POSITION, (1, 1, 1, 0))

        self.font = pygame.font.SysFont('Arial', 18)
        self.colors = {
            'beginner': (0.2, 0.2, 0.8), 'basic': (0.2, 0.8, 0.2),
            'intermediate': (0.8, 0.8, 0.2), 'advanced': (0.8, 0.4, 0.2),
            'fluent': (0.8, 0.2, 0.2), 'vocabulary': (0.7, 0.7, 1.0),
            'conversation': (1.0, 0.7, 0.7), 'grammar': (0.7, 1.0, 0.7),
            'culture': (1.0, 0.7, 1.0), 'agent': (1.0, 1.0, 0.0)
        }

    def draw_text(self, text, position, color=(1, 1, 1)):
        text_surface = self.font.render(
            text, True, (int(color[0]*255), int(color[1]*255), int(color[2]*255)))
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        text_width, text_height = text_surface.get_size()

        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_width,
                     text_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glPushMatrix()
        glTranslatef(position[0], position[1], position[2])
        scale_factor = 0.01

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex3f(0, 0, 0)
        glTexCoord2f(1, 0)
        glVertex3f(text_width * scale_factor, 0, 0)
        glTexCoord2f(1, 1)
        glVertex3f(text_width * scale_factor, text_height * scale_factor, 0)
        glTexCoord2f(0, 1)
        glVertex3f(0, text_height * scale_factor, 0)
        glEnd()

        glDisable(GL_BLEND)
        glDisable(GL_TEXTURE_2D)
        glPopMatrix()
        glDeleteTextures(1, [texture_id])

    def draw_cube(self, position, size, color, label=None):
        glPushMatrix()
        glTranslatef(position[0], position[1], position[2])
        glColor3f(*color)

        vertices = [
            (size/2, size/2, size/2), (size/2, -size/2, size/2),
            (-size/2, -size/2, size/2), (-size/2, size/2, size/2),
            (size/2, size/2, -size/2), (size/2, -size/2, -size/2),
            (-size/2, -size/2, -size/2), (-size/2, size/2, -size/2)
        ]
        surfaces = [
            (0, 1, 2, 3), (4, 5, 6, 7), (0, 4, 7, 3),
            (1, 5, 6, 2), (0, 4, 5, 1), (3, 7, 6, 2)
        ]

        glBegin(GL_QUADS)
        for surface in surfaces:
            for vertex in surface:
                glVertex3fv(vertices[vertex])
        glEnd()

        glColor3f(0.0, 0.0, 0.0)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                 (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()

        if label:
            self.draw_text(label, (-size/3, 0, size/2 + 0.05), (0, 0, 0))
        glPopMatrix()

    def draw_agent(self, position, size, level):
        glPushMatrix()
        glTranslatef(position[0], position[1], position[2])
        level_colors = [
            self.colors['beginner'], self.colors['basic'], self.colors['intermediate'],
            self.colors['advanced'], self.colors['fluent']
        ]
        glColor3f(*level_colors[level])
        quadric = gluNewQuadric()
        gluSphere(quadric, size, 32, 32)
        gluDeleteQuadric(quadric)
        glPopMatrix()

    def draw_proficiency_path(self, current_level):
        levels = [
            {"name": "Beginner", "position": (-4, 0, 0)},
            {"name": "Basic", "position": (-2, 0, 0)},
            {"name": "Intermediate", "position": (0, 0, 0)},
            {"name": "Advanced", "position": (2, 0, 0)},
            {"name": "Fluent", "position": (4, 0, 0)}
        ]
        glBegin(GL_LINE_STRIP)
        glColor3f(0.5, 0.5, 0.5)
        for level in levels:
            glVertex3fv(level["position"])
        glEnd()

        for i, level in enumerate(levels):
            color = (0.8, 0.8, 0.2) if i == current_level else (0.5, 0.5, 0.5)
            size = 1.0 if i == current_level else 0.8
            self.draw_cube(level["position"], size, color, level["name"])

    def draw_action_elements(self, last_action=None):
        actions = [
            {"name": "Vocabulary",
                "position": (-2, 2, 0), "color": self.colors['vocabulary']},
            {"name": "Conversation", "position": (
                2, 2, 0), "color": self.colors['conversation']},
            {"name": "Grammar",
                "position": (-2, -2, 0), "color": self.colors['grammar']},
            {"name": "Culture", "position": (
                2, -2, 0), "color": self.colors['culture']}
        ]
        for i, action in enumerate(actions):
            size = 1.0 if last_action == i else 0.8
            self.draw_cube(action["position"], size,
                           action["color"], action["name"])

    def draw_stats(self, performance, engagement, reward):
        stats_position = (-4, -3, 0)
        glPushMatrix()
        glTranslatef(stats_position[0], stats_position[1], stats_position[2])
        glColor4f(0.2, 0.2, 0.2, 0.7)
        glBegin(GL_QUADS)
        glVertex3f(0, 0, 0)
        glVertex3f(8, 0, 0)
        glVertex3f(8, 1.5, 0)
        glVertex3f(0, 1.5, 0)
        glEnd()
        glPopMatrix()

        self.draw_stat_bar(stats_position, "Performance",
                           performance, (0.2, 0.6, 1.0))
        self.draw_stat_bar((stats_position[0], stats_position[1] + 0.5, stats_position[2]),
                           "Engagement", engagement, (1.0, 0.6, 0.2))
        self.draw_stat_bar((stats_position[0], stats_position[1] + 1.0, stats_position[2]),
                           "Reward", min(100, max(0, reward + 50)), (0.6, 1.0, 0.2))

    def draw_stat_bar(self, position, label, value, color):
        self.draw_text(f"{label}: {value:.1f}",
                       (position[0] + 0.1, position[1] + 0.1, position[2] + 0.01), (1, 1, 1))
        glPushMatrix()
        glTranslatef(position[0] + 3, position[1] + 0.25, position[2] + 0.01)
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_QUADS)
        glVertex3f(0, 0, 0)
        glVertex3f(4.5, 0, 0)
        glVertex3f(4.5, 0.2, 0)
        glVertex3f(0, 0.2, 0)
        glEnd()
        fill_width = (value / 100) * 4.5
        glTranslatef(0, 0, 0.01)
        glColor3f(*color)
        glBegin(GL_QUADS)
        glVertex3f(0, 0, 0)
        glVertex3f(fill_width, 0, 0)
        glVertex3f(fill_width, 0.2, 0)
        glVertex3f(0, 0.2, 0)
        glEnd()
        glPopMatrix()

    def render_dynamic_scene(self, current_level=0, position=(-4, 0, 0.5), performance=50, engagement=70, reward=0, last_action=None):
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.draw_proficiency_path(current_level)
        self.draw_action_elements(last_action)
        self.draw_agent(position, 0.4, current_level)
        self.draw_stats(performance, engagement, reward)
        self.draw_text("Kinyarwanda Language Learning",
                       (-3.5, 3, 0), (1, 0, 0))
        glFlush()
        pygame.display.flip()

    def render_static_scene(self, current_level=0, position=(-4, 0, 0.5), performance=50, engagement=70, reward=0, last_action=None):
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.draw_proficiency_path(current_level)
        self.draw_action_elements(last_action)
        self.draw_agent(position, 0.4, current_level)
        self.draw_stats(performance, engagement, reward)
        self.draw_text(
            "Kinyarwanda Language Learning - Static View", (-3.5, 3, 0), (1, 0, 0))
        glFlush()
        pygame.display.flip()

    def save_screenshot(self, filename="temp.png", return_array=False):
        buffer = glReadPixels(0, 0, self.window_width,
                              self.window_height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(buffer, dtype=np.uint8).reshape(
            self.window_height, self.window_width, 3)
        image = np.flipud(image)
        if return_array:
            return image
        pygame.image.save(pygame.surfarray.make_surface(
            np.transpose(image, (1, 0, 2))), filename)
        return None

    def close(self):
        pygame.quit()

    @staticmethod
    def render_static_video(output_file="video/static_visualization.mp4", duration=5):
        """Render a static scene as a video and save it in the video folder."""
        renderer = LanguageLearningRenderer(800, 600)
        fps = 30
        total_frames = duration * fps  # e.g., 5 seconds * 30 FPS = 150 frames
        frames = []

        # Static parameters for visualization
        current_level = 2  # Intermediate level
        position = (0, 0, 0.5)  # Agent at Intermediate position
        performance = 75.0
        engagement = 80.0
        reward = 50.0
        last_action = 1  # Highlight Conversation

        # Ensure video folder exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        print("Rendering static video...")
        for frame_num in range(total_frames):
            # Handle events to keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    renderer.close()
                    return

            # Render the static scene
            renderer.render_static_scene(
                current_level=current_level,
                position=position,
                performance=performance,
                engagement=engagement,
                reward=reward,
                last_action=last_action
            )

            # Capture frame
            frame = renderer.save_screenshot(return_array=True)
            if frame is not None:
                frames.append(frame)

            # Small delay to simulate real-time rendering (optional)
            pygame.time.wait(1000 // fps)  # Approx 33ms for 30 FPS

        # Save frames as video
        if frames:
            writer = imageio.get_writer(output_file, fps=fps)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            print(
                f"Static video saved as {output_file} with {len(frames)} frames (~{len(frames)/fps:.1f} seconds)")
        else:
            print("Error: No frames captured for video.")

        renderer.close()


if __name__ == "__main__":
    # Run the static video rendering when the script is executed directly
    output_file = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), "video", "static_visualization.mp4")
    LanguageLearningRenderer.render_static_video(output_file)
