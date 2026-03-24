import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

prev_landmarks = None

with open("gesture_data.csv", "a", newline="") as f:
    writer = csv.writer(f)

    print("\n1 → infinite_void")
    print("2 → hollow_purple_charge")
    print("3 → hollow_purple_release")
    print("0 → idle")
    print("q → quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        key = cv2.waitKey(1) & 0xFF

        label = None
        if key == ord('1'):
            label = "infinite_void"
        elif key == ord('2'):
            label = "hollow_purple_charge"
        elif key == ord('3'):
            label = "hollow_purple_release"
        elif key == ord('0'):
            label = "idle"

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                current = [(lm.x, lm.y) for lm in hand.landmark]

                data = []

                # velocity
                if prev_landmarks:
                    for (x, y), (px, py) in zip(current, prev_landmarks):
                        data.extend([x - px, y - py])
                else:
                    data.extend([0] * (21 * 2))

                # relative position (wrist-based)
                wrist_x, wrist_y = current[0]
                for (x, y) in current:
                    data.extend([x - wrist_x, y - wrist_y])

                prev_landmarks = current

                if label:
                    writer.writerow(data + [label])
                    print(f"Saved: {label}")

        cv2.imshow("Collect Data", frame)

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()