from collections import deque, Counter

class GestureStabilizer:
    def __init__(self, buffer_size=10, threshold=7, cooldown_frames=15):
        self.buffer = deque(maxlen=buffer_size)
        self.threshold = threshold
        self.cooldown = 0
        self.cooldown_frames = cooldown_frames
        self.current = "idle"

    def update(self, gesture):
        self.buffer.append(gesture)

        if self.cooldown > 0:
            self.cooldown -= 1
            return self.current

        if len(self.buffer) == self.buffer.maxlen:
            most_common, count = Counter(self.buffer).most_common(1)[0]

            if count >= self.threshold and most_common != self.current:
                self.current = most_common
                self.cooldown = self.cooldown_frames

        return self.current