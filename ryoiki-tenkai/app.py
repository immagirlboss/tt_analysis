import cv2
import mediapipe as mp
import pickle
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import random
import math
from collections import deque, Counter


with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
h, w = 480, 640
ANIM_DURATION = 90
current_gesture = "idle"
prev_gesture = "idle"
anim_frame = 0

# particle system
particles = []

def add_particles(cx, cy, color, count=15, speed=8):
    for _ in range(count):
        angle = random.uniform(0, 2 * math.pi)
        spd = random.uniform(2, speed)
        particles.append({
            "x": float(cx), "y": float(cy),
            "vx": math.cos(angle) * spd,
            "vy": math.sin(angle) * spd,
            "color": color,
            "life": random.randint(15, 35),
            "size": random.randint(2, 6)
        })

def update_particles(frame):
    dead = []
    for p in particles:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["life"] -= 1
        alpha = max(0, p["life"] / 35)
        c = tuple(int(ch * alpha) for ch in p["color"])
        cx, cy = int(p["x"]), int(p["y"])
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(frame, (cx, cy), p["size"], c, -1)
        if p["life"] <= 0:
            dead.append(p)
    for p in dead:
        particles.remove(p)

def draw_speed_lines(frame, cx, cy, color, count=40, length_range=(80, 250)):
    overlay = frame.copy()
    for _ in range(count):
        angle = random.uniform(0, 2 * math.pi)
        length = random.randint(*length_range)
        thickness = random.randint(1, 3)
        start = (
            int(cx + math.cos(angle) * random.randint(10, 40)),
            int(cy + math.sin(angle) * random.randint(10, 40))
        )
        end = (
            int(cx + math.cos(angle) * length),
            int(cy + math.sin(angle) * length)
        )
        cv2.line(overlay, start, end, color, thickness)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

def draw_electric(frame, cx, cy, color, radius, bolts=8):
    for _ in range(bolts):
        angle = random.uniform(0, 2 * math.pi)
        pts = [(cx, cy)]
        r = 0
        while r < radius:
            r += random.randint(10, 25)
            jitter_angle = angle + random.uniform(-0.4, 0.4)
            px = int(cx + math.cos(jitter_angle) * r)
            py = int(cy + math.sin(jitter_angle) * r)
            pts.append((px, py))
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i+1], color, random.randint(1, 3))

def draw_ink_burst(frame, cx, cy, t):
    """Dark ink explosion for infinite void"""
    overlay = frame.copy()
    # dark void core
    radius = int(30 + t * 2.5)
    cv2.circle(overlay, (cx, cy), radius, (20, 0, 30), -1)
    # ink splatter blobs
    random.seed(42)
    for _ in range(12):
        angle = random.uniform(0, 2 * math.pi)
        dist = random.randint(radius // 2, radius + int(t * 1.5))
        bx = int(cx + math.cos(angle) * dist)
        by = int(cy + math.sin(angle) * dist)
        blob_r = random.randint(8, 25)
        cv2.ellipse(overlay, (bx, by),
                    (blob_r, blob_r // 2),
                    int(math.degrees(angle)), 0, 360,
                    (15, 0, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

def put_japanese_text(frame, text, pos, size=48, color=(200, 0, 255)):
    """Render Japanese text using PIL"""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        # try system fonts that support Japanese
        font = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB W3.otf", size)
        except:
            font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=(color[0], color[1], color[2]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ── INFINITE VOID ─────────────────────────────────────────────
def draw_infinite_void(frame, t):
    cx, cy = w // 2, h // 2

    # darken whole frame gradually
    darkness = min(0.7, t / ANIM_DURATION * 0.9)
    dark_overlay = np.zeros_like(frame)
    cv2.addWeighted(dark_overlay, darkness, frame, 1 - darkness, 0, frame)

    # ink burst at center
    draw_ink_burst(frame, cx, cy, t)

    # speed lines radiating out — white/purple/blue mix
    if t % 3 == 0:
        draw_speed_lines(frame, cx, cy, (255, 255, 255), count=30, length_range=(60, 280))
        draw_speed_lines(frame, cx, cy, (180, 0, 255), count=20, length_range=(40, 200))
        draw_speed_lines(frame, cx, cy, (100, 50, 255), count=15, length_range=(30, 160))

    # particles
    if t % 2 == 0:
        add_particles(cx, cy, (180, 0, 255), count=8, speed=6)
        add_particles(cx, cy, (255, 255, 255), count=4, speed=10)
    update_particles(frame)

    # Japanese text
    if t > 20:
        frame = put_japanese_text(frame, "無量空処", (cx - 80, cy + 120), size=52, color=(200, 0, 255))
        frame = put_japanese_text(frame, "INFINITE VOID", (cx - 95, cy + 175), size=32, color=(255, 255, 255))

    return frame

# ── BLUE + RED (CHARGE) ───────────────────────────────────────
def draw_hollow_purple_charge(frame, t):
    pulse = int(15 + t * 0.8)

    # ── BLUE orb (left side) ──
    bcx, bcy = 100, h // 2
    overlay = frame.copy()
    cv2.circle(overlay, (bcx, bcy), pulse + 20, (255, 80, 0), -1)   # glow
    cv2.circle(overlay, (bcx, bcy), pulse, (255, 200, 50), -1)       # bright core
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    draw_electric(frame, bcx, bcy, (255, 150, 50), pulse + 30)
    if t % 3 == 0:
        add_particles(bcx, bcy, (255, 100, 0), count=5, speed=4)

    # ── RED orb (right side) ──
    rcx, rcy = w - 100, h // 2
    overlay2 = frame.copy()
    cv2.circle(overlay2, (rcx, rcy), pulse + 20, (0, 0, 180), -1)   # glow
    cv2.circle(overlay2, (rcx, rcy), pulse, (0, 50, 255), -1)        # bright core
    cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0, frame)
    draw_electric(frame, rcx, rcy, (0, 0, 255), pulse + 30)

    # dark smoke clouds around red
    smoke_overlay = frame.copy()
    for i in range(5):
        sx = rcx + random.randint(-60, 60)
        sy = rcy + random.randint(-60, 60)
        cv2.circle(smoke_overlay, (sx, sy), random.randint(15, 35), (0, 0, 40), -1)
    cv2.addWeighted(smoke_overlay, 0.3, frame, 0.7, 0, frame)

    if t % 3 == 0:
        add_particles(rcx, rcy, (0, 0, 255), count=5, speed=4)

    update_particles(frame)

    frame = put_japanese_text(frame, "蒼", (bcx - 20, bcy - pulse - 60), size=48, color=(255, 150, 50))
    frame = put_japanese_text(frame, "BLUE", (bcx - 25, bcy - pulse - 20), size=28, color=(255, 200, 100))
    frame = put_japanese_text(frame, "赫", (rcx - 20, rcy - pulse - 60), size=48, color=(0, 50, 255))
    frame = put_japanese_text(frame, "RED", (rcx - 25, rcy - pulse - 20), size=28, color=(100, 100, 255))

    return frame

# ── HOLLOW PURPLE (RELEASE) ───────────────────────────────────
def draw_hollow_purple_release(frame, t):
    cx, cy = w // 2, h // 2
    radius = int(20 + t * 3.5)

    # flash white at start
    if t < 10:
        white = np.ones_like(frame) * 255
        alpha = 1.0 - (t / 10)
        cv2.addWeighted(white, alpha, frame, 1 - alpha, 0, frame)

    # expanding purple burst
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), radius, (180, 0, 220), -1)
    cv2.circle(overlay, (cx, cy), int(radius * 0.6), (220, 100, 255), -1)
    cv2.circle(overlay, (cx, cy), int(radius * 0.3), (255, 220, 255), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # purple lightning outward
    draw_electric(frame, cx, cy, (200, 50, 255), radius + 40, bolts=12)
    draw_electric(frame, cx, cy, (255, 150, 255), radius + 20, bolts=8)

    # speed lines
    if t % 2 == 0:
        draw_speed_lines(frame, cx, cy, (200, 0, 255), count=25, length_range=(radius, radius + 150))

    if t % 2 == 0:
        add_particles(cx, cy, (180, 0, 255), count=10, speed=9)
        add_particles(cx, cy, (255, 200, 255), count=5, speed=12)
    update_particles(frame)

    if t > 15:
        frame = put_japanese_text(frame, "虚式「茈」", (cx - 90, cy + radius + 20), size=44, color=(220, 100, 255))
        frame = put_japanese_text(frame, "HOLLOW PURPLE", (cx - 105, cy + radius + 70), size=28, color=(255, 255, 255))

    return frame

# ── MAIN LOOP ─────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (w, h))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = "idle"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks_flat = []
            for lm in hand_landmarks.landmark:
                landmarks_flat.extend([lm.x, lm.y])
            gesture = model.predict([landmarks_flat])[0]

    if gesture != "idle" and gesture != prev_gesture:
        current_gesture = gesture
        anim_frame = 0
        particles.clear()

    prev_gesture = gesture

    if current_gesture != "idle" and anim_frame < ANIM_DURATION:
        if current_gesture == "infinite_void":
            frame = draw_infinite_void(frame, anim_frame)
        elif current_gesture == "hollow_purple_charge":
            frame = draw_hollow_purple_charge(frame, anim_frame)
        elif current_gesture == "hollow_purple_release":
            frame = draw_hollow_purple_release(frame, anim_frame)
        anim_frame += 1
    elif anim_frame >= ANIM_DURATION:
        current_gesture = "idle"
        particles.clear()

    color = (0, 255, 0) if gesture == "idle" else (180, 0, 255)
    cv2.putText(frame, gesture, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Ryoiki Tenkai", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()