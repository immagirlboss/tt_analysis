import cv2
import mediapipe as mp
import csv
import os
import time

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=0
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

CSV_FILE = "gesture_data.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"{axis}{i}" for i in range(21) for axis in ["x", "y"]] + ["label"]
        writer.writerow(header)

LABELS = {
    ord('0'): "idle",
    ord('1'): "infinite_void",
    ord('2'): "hollow_purple_charge",
    ord('3'): "hollow_purple_release",
}

COLORS = {
    "idle":                  (180, 180, 180),
    "infinite_void":         (255, 0,   200),
    "hollow_purple_charge":  (255, 150, 0  ),
    "hollow_purple_release": (200, 0,   255),
}

current_label = "idle"
sample_count  = 0
HOLD_TIME     = 2.0
BURST_COUNT   = 30
holding_since = None
burst_done    = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF

    if key in LABELS:
        current_label = LABELS[key]
        holding_since = None
        burst_done    = False
        print(f"Switched to: {current_label}")

    color = COLORS.get(current_label, (255, 255, 255))
    cv2.rectangle(frame, (0, 0), (640, 110), (20, 20, 20), -1)
    cv2.putText(frame, f"Label: {current_label}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Total samples: {sample_count}", (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "0=idle 1=void 2=charge 3=release | S=save | Q=quit",
                (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks_flat = []
            for lm in hand_landmarks.landmark:
                landmarks_flat.extend([lm.x, lm.y])

            def save_sample():
                global sample_count
                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(landmarks_flat + [current_label])
                sample_count += 1

            # manual save
            if key == ord('s'):
                save_sample()
                print(f"Saved sample {sample_count} for '{current_label}'")

            # auto-save on hold
            if current_label != "idle":
                if holding_since is None:
                    holding_since = time.time()
                    burst_done    = False

                elapsed = time.time() - holding_since
                bar_w   = min(int((elapsed / HOLD_TIME) * 620), 620)
                cv2.rectangle(frame, (10, 98), (10 + bar_w, 108), color, -1)

                if elapsed >= HOLD_TIME and not burst_done:
                    for _ in range(BURST_COUNT):
                        save_sample()
                    burst_done    = True
                    holding_since = None
                    print(f"Auto-saved {BURST_COUNT} samples for '{current_label}' (total: {sample_count})")
            else:
                holding_since = None

    cv2.imshow("Data Collector — Ryoiki Tenkai", frame)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nDone. Total samples: {sample_count}")