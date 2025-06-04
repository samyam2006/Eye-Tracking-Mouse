import time
import numpy as np
import pyautogui

class DwellClickDetector:
    def __init__(self, dwell_time=2.0, move_tolerance=40):
        self.dwell_time = dwell_time
        self.move_tolerance = move_tolerance
        self.last_position = None
        self.enter_time = None

    def update(self, current_pos):
        if self.last_position is None:
            self.last_position = current_pos
            self.enter_time = time.time()
            return

        dist = np.linalg.norm(np.array(current_pos) - np.array(self.last_position))

        if dist < self.move_tolerance:
            if time.time() - self.enter_time > self.dwell_time:
                print("🖱️ Dwell click triggered!")
                pyautogui.click()
                self.enter_time = time.time() + 1  # Add buffer to prevent double clicks
        else:
            self.last_position = current_pos
            self.enter_time = time.time()
