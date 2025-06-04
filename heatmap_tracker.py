import matplotlib.pyplot as plt
import numpy as np
import time
import os

class GazeHeatmap:
    def __init__(self, screen_w, screen_h, max_points=5000):
        self.points = []
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.max_points = max_points

    def log(self, x, y):
        if 0 <= x <= self.screen_w and 0 <= y <= self.screen_h:
            self.points.append((x, y))
        if len(self.points) > self.max_points:
            self.points.pop(0)

    def save_heatmap(self, filename=None):
        if not self.points:
            print("No gaze points logged.")
            return

        x, y = zip(*self.points)
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=(100, 100),
                                                 range=[[0, self.screen_w], [0, self.screen_h]])
        heatmap = np.rot90(heatmap)
        heatmap = np.flipud(heatmap)

        plt.imshow(heatmap, extent=[0, self.screen_w, 0, self.screen_h], cmap='jet', alpha=0.8)
        plt.title("Gaze Heatmap")
        plt.xlabel("Screen X")
        plt.ylabel("Screen Y")

        if not filename:
            filename = f"assets/gaze_heatmap_{int(time.time())}.png"

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()
        print(f"📊 Heatmap saved to {filename}")
