import cv2
import numpy as np
import random
import math

particles = []

# ───── PARTICLES ─────
def add_particles(cx, cy, color, count=8, speed=6):
    for _ in range(count):
        angle = random.uniform(0, 2 * math.pi)
        particles.append({
            "x": cx,
            "y": cy,
            "vx": math.cos(angle) * random.uniform(1, speed),
            "vy": math.sin(angle) * random.uniform(1, speed),
            "life": random.randint(15, 30),
            "color": color
        })


def update_particles(frame):
    global particles
    new_particles = []

    for p in particles:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["life"] -= 1

        if p["life"] > 0:
            alpha = p["life"] / 30
            color = tuple(int(c * alpha) for c in p["color"])
            cv2.circle(frame, (int(p["x"]), int(p["y"])), 3, color, -1)
            new_particles.append(p)

    particles = new_particles


def clear_particles():
    global particles
    particles = []


# ───── VISUAL FX ─────
def add_glow(frame, intensity=10):
    blur = cv2.GaussianBlur(frame, (0, 0), intensity)
    return cv2.addWeighted(blur, 0.6, frame, 0.4, 0)


def camera_shake(frame, intensity=8):
    h, w = frame.shape[:2]
    dx = random.randint(-intensity, intensity)
    dy = random.randint(-intensity, intensity)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (w, h))


def screen_flash(frame, strength):
    white = np.ones_like(frame) * 255
    return cv2.addWeighted(white, strength, frame, 1-strength, 0)


def chromatic_aberration(frame, shift=3):
    b, g, r = cv2.split(frame)

    M1 = np.float32([[1, 0, shift], [0, 1, 0]])
    M2 = np.float32([[1, 0, -shift], [0, 1, 0]])

    r = cv2.warpAffine(r, M1, (frame.shape[1], frame.shape[0]))
    b = cv2.warpAffine(b, M2, (frame.shape[1], frame.shape[0]))

    return cv2.merge([b, g, r])


def draw_energy_flow(frame, cx, cy, dx, dy, color):
    for i in range(20):
        t = i / 20
        x = int(cx + dx * t * 300 + random.randint(-5, 5))
        y = int(cy + dy * t * 300 + random.randint(-5, 5))
        size = int(8 * (1 - t))
        cv2.circle(frame, (x, y), size, color, -1)


# ───── ANIMATIONS ─────
def hollow_purple_charge(frame, progress, w, h):
    cy = h // 2

    # blue → left
    draw_energy_flow(frame, 100, cy, 1, 0, (255, 150, 50))

    # red → right
    draw_energy_flow(frame, w-100, cy, -1, 0, (0, 50, 255))

    if random.random() < 0.4:
        add_particles(100, cy, (255,150,50), 4)
        add_particles(w-100, cy, (0,50,255), 4)

    update_particles(frame)
    return add_glow(frame, 8)


def hollow_purple_release(frame, progress, w, h):
    cx, cy = w//2, h//2
    radius = int(30 + progress * 300)

    overlay = frame.copy()
    for i in range(3):
        r = int(radius * (1 + i*0.3))
        alpha = 0.2 / (i+1)
        cv2.circle(overlay, (cx, cy), r, (200,0,255), -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

    if progress < 0.1:
        frame = screen_flash(frame, 0.8)

    if progress > 0.6:
        frame = camera_shake(frame, 8)
        frame = chromatic_aberration(frame, 4)

    if random.random() < 0.5:
        add_particles(cx, cy, (200,0,255), 10)

    update_particles(frame)
    return add_glow(frame, 12)


def infinite_void(frame, progress, w, h):
    cx, cy = w//2, h//2

    overlay = np.zeros_like(frame)
    frame = cv2.addWeighted(frame, 1-progress*0.8, overlay, progress*0.8, 0)

    if random.random() < 0.3:
        add_particles(cx, cy, (180,0,255), 6)

    update_particles(frame)
    return add_glow(frame, 10)