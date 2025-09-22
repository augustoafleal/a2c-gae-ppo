import imageio
import os
from datetime import datetime


class RenderRecorder:
    def __init__(self, filename=None, fps=4):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video/video_{timestamp}.mp4"

        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        self.filename = filename
        self.fps = fps
        self.frames = []

    def capture(self, frame):
        self.frames.append(frame)

    def save(self):
        imageio.mimsave(self.filename, self.frames, fps=self.fps)
        print(f"Video saved in {self.filename}")

    def reset(self):
        self.frames = []
