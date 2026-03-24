import random
from effects import *

ANIM_DURATION = 90

def ease_out(t, d):
    x = t / d
    return 1 - (1 - x) ** 3


class AnimationEngine:
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.current = "idle"
        self.frame = 0
        self.seed = random.randint(0, 99999)

    def trigger(self, gesture):
        if gesture != self.current:
            self.current = gesture
            self.frame = 0
            self.seed = random.randint(0, 99999)
            clear_particles()

    def update(self, frame):
        if self.current == "idle":
            return frame

        t = self.frame
        progress = ease_out(t, ANIM_DURATION)

        random.seed(self.seed + t)

        if self.current == "infinite_void":
            frame = infinite_void(frame, progress, self.w, self.h)

        elif self.current == "hollow_purple_charge":
            frame = hollow_purple_charge(frame, progress, self.w, self.h)

        elif self.current == "hollow_purple_release":
            frame = hollow_purple_release(frame, progress, self.w, self.h)

        self.frame += 1

        if self.frame > ANIM_DURATION:
            self.current = "idle"
            clear_particles()

        return frame