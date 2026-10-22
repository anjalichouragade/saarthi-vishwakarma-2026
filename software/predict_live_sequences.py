import time
from collections import deque
import cv2
import joblib
import numpy as np
import mediapipe as mp
from features import extract_features, sequence_features

MODEL_PATH = "software/sign_sequence_model.joblib"
SEQ_LEN = 30
WINDOW_SECONDS = 1.5   # must match RECORD_SECONDS used when recording
THRESHOLD = 0.50       # below this confidence -> "Please repeat"
VOTES_NEEDED = 3       # same answer this many times in a row
HOLD_SECONDS = 2.0     # how long a recognised word stays on screen

model = joblib.load(MODEL_PATH)
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

buffer = deque()                      # (timestamp, features, hand_seen)
recent = deque(maxlen=VOTES_NEEDED)
text, colour, shown_until = "Waiting for sign...", (200, 200, 200), 0.0
last_predict = 0.0

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera could not be opened.")
    raise SystemExit

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame received from camera")
            break
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        display = cv2.flip(frame, 1)

        now = time.time()
        hand_seen = bool(results.left_hand_landmarks or results.right_hand_landmarks)
        buffer.append((now, extract_features(results), hand_seen))
        while buffer and now - buffer[0][0] > WINDOW_SECONDS:
            buffer.popleft()

        full_window = len(buffer) >= 10 and (buffer[-1][0] - buffer[0][0]) >= 0.9 * WINDOW_SECONDS
        if full_window and now - last_predict > 0.3:
            last_predict = now
            feats = np.array([b[1] for b in buffer])
            idx = np.linspace(0, len(feats) - 1, SEQ_LEN).round().astype(int)
            hand_ratio = np.mean([b[2] for b in buffer])

            if hand_ratio < 0.5:
                recent.clear()
            else:
                probs = model.predict_proba([sequence_features(feats[idx])])[0]
                best = probs.argmax()
                label = model.classes_[best] if probs[best] >= THRESHOLD else "LOW"
                recent.append((label, probs[best]))

                if len(recent) == VOTES_NEEDED and len({r[0] for r in recent}) == 1:
                    if label == "LOW":
                        text, colour, shown_until = "Please repeat", (0, 165, 255), now + 1.0
                    elif label != "NONE":
                        text, colour = f"{label} ({probs[best]:.0%})", (0, 255, 0)
                        shown_until = now + HOLD_SECONDS
                    recent.clear()

        if now > shown_until:
            text, colour = "Waiting for sign...", (200, 200, 200)

        cv2.putText(display, text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)
        cv2.imshow("SAARTHI - live sign recognition", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()