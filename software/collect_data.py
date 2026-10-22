import csv
import os
import cv2
import mediapipe as mp

LABELS = {"1": "HELP", "2": "WATER", "3": "YES", "4": "NO", "5": "DOCTOR"}
OUT = "data/landmarks.csv"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def normalise(hand):
    """Make landmarks independent of hand position and size in the frame."""
    pts = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
    wx, wy, wz = pts[0]  # wrist is landmark 0
    pts = [(x - wx, y - wy, z - wz) for x, y, z in pts]
    scale = max(max(abs(x), abs(y)) for x, y, _ in pts) or 1
    flat = []
    for x, y, z in pts:
        flat += [x / scale, y / scale, z / scale]
    return flat


os.makedirs("data", exist_ok=True)
new_file = not os.path.exists(OUT)
counts = {name: 0 for name in LABELS.values()}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera could not be opened.")
    raise SystemExit

with open(OUT, "a", newline="") as f, mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6
) as hands:
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["label"] + [f"{a}{i}" for i in range(21) for a in "xyz"])

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame received from camera")
            break
        frame = cv2.flip(frame, 1)
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        hand = None
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if chr(key) in LABELS and hand is not None:
            name = LABELS[chr(key)]
            writer.writerow([name] + normalise(hand))
            f.flush()
            counts[name] += 1

        y = 25
        for k, name in LABELS.items():
            cv2.putText(frame, f"{k}: {name} ({counts[name]})", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25
        cv2.imshow("SAARTHI - collect data", frame)

cap.release()
cv2.destroyAllWindows()
