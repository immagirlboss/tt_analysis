import cv2
import mediapipe as mp
import pickle
import threading
import numpy as np
from collections import deque, Counter
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ryoiki'

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=False,
    engineio_logger=False
)

with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=0
)

# ── Tuned for speed ───────────────────────────
SMOOTHING_WINDOW  = 3
CONFIDENCE_THRESH = 0.30
COOLDOWN_FRAMES   = 30

@app.route("/")
def index():
    return render_template("index.html")

def detection_loop():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    gesture_buffer = deque(maxlen=SMOOTHING_WINDOW)
    last_sent      = "idle"
    cooldown       = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        raw_gesture = "idle"
        confidence  = 1.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm_flat = []
                for lm in hand_landmarks.landmark:
                    lm_flat.extend([lm.x, lm.y])

                proba       = model.predict_proba([lm_flat])[0]
                best_idx    = np.argmax(proba)
                confidence  = proba[best_idx]
                raw_gesture = model.classes_[best_idx]

                if confidence < CONFIDENCE_THRESH:
                    raw_gesture = "idle"

        gesture_buffer.append(raw_gesture)

        if len(gesture_buffer) == SMOOTHING_WINDOW:
            counts = Counter(gesture_buffer)
            top, freq = counts.most_common(1)[0]
            stable = top if (top != "idle" and freq >= 2) else "idle"
        else:
            stable = "idle"

        if cooldown > 0:
            cooldown -= 1

        if stable != last_sent:
            if stable == "idle" or cooldown == 0:
                print(f"→ {stable} ({confidence:.2f})")
                # short key names = less bytes over socket = faster
                socketio.emit("g", {"n": stable, "c": round(float(confidence), 2)})
                last_sent = stable
                if stable != "idle":
                    cooldown = COOLDOWN_FRAMES

    cap.release()

if __name__ == "__main__":
    threading.Thread(target=detection_loop, daemon=True).start()
    print("→ http://127.0.0.1:5001")
    socketio.run(
        app, host='127.0.0.1', port=5001,
        debug=False, use_reloader=False,
        allow_unsafe_werkzeug=True
    )
