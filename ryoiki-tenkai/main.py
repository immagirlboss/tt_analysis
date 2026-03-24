import cv2
import mediapipe as mp
import pickle

from gesture import GestureStabilizer
from animation import AnimationEngine

# load model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

stabilizer = GestureStabilizer()
engine = AnimationEngine(640, 480)

frame_count = 0
gesture = "idle"
prev_landmarks = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    small = cv2.resize(frame, (320, 240))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    # run model every 2 frames
    if frame_count % 2 == 0:
        gesture_pred = "idle"

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                current = [(lm.x, lm.y) for lm in hand.landmark]

                data = []

                # ── velocity (movement between frames) ──
                if prev_landmarks:
                    for (x, y), (px, py) in zip(current, prev_landmarks):
                        data.extend([x - px, y - py])
                else:
                    data.extend([0] * (21 * 2))

                # ── relative position (shape, centered at wrist) ──
                wrist_x, wrist_y = current[0]
                for (x, y) in current:
                    data.extend([x - wrist_x, y - wrist_y])

                prev_landmarks = current

                # ── prediction with confidence ──
                probs = model.predict_proba([data])[0]
                confidence = max(probs)

                if confidence > 0.8:
                    gesture_pred = le.inverse_transform([probs.argmax()])[0]
                else:
                    gesture_pred = "idle"

                gesture = stabilizer.update(gesture_pred)

    frame_count += 1

    engine.trigger(gesture)
    frame = engine.update(frame)

    cv2.putText(frame, gesture, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 0, 255), 2)

    cv2.imshow("Domain Expansion", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()