import gymnasium as gym
import cv2
import numpy as np


class ContinuousVideoRecorder:
    def __init__(self, env, output_path="continuous_video.mp4", fps=30):
        self.env = env
        self.output_path = output_path
        self.fps = fps
        self.video_writer = None

    def start(self):
        # Initialize video writer with the first frame's dimensions
        frame = self.env.render()
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
        self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))

    def step(self, action):
        # Step the environment and write the frame
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self.env.render()
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV
        return obs, reward, terminated, truncated, info

    def reset(self):
        # Reset the environment and write the frame
        obs = self.env.reset()
        frame = self.env.render()
        if self.video_writer is None:
            self.start()
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return obs

    def close(self):
        # Release the video writer
        if self.video_writer is not None:
            self.video_writer.release()
        self.env.close()
