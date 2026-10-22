import os
import time
import cv2
import numpy as np
import mediapipe as mp
from features import extract_features

LABELS = {"1": "STOP", "2": "WATER", "3": "YES", "4": "NO", "5": "DOCTOR", "6": "NONE", "7": "HELP"}
SEQ_LEN = 30
COUNTDOWN = 1.5        # seconds to get ready
RECORD_SECONDS = 1.5   # length of one sign
OUT_DIR = "data/sequences"

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

for name in LABELS.values():
    os.makedirs(os.path.join(OUT_DIR, name), exist_ok=True)


def count_files(name):
    return len([f for f in os.listdir(os.path.join(OUT_DIR, name)) if f.endswith(".npy")])


counts = {name: count_files(name) for name in LABELS.values()}


def resample(frames):
    frames = np.array(frames)
    idx = np.linspace(0, len(frames) - 1, SEQ_LEN).round().astype(int)
    return frames[idx]


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera could not be opened.")
    raise SystemExit

state, current, t0, frames, hand_frames = "idle", None, 0.0, [], 0
message = "Press a key (1-6) to record"

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
        if state == "countdown":
            message = f"GET READY: {current}"
            if now - t0 >= COUNTDOWN:
                state, t0, frames, hand_frames = "recording", now, [], 0
        elif state == "recording":
            message = f"RECORDING: {current}"
            frames.append(extract_features(results))
            if results.left_hand_landmarks or results.right_hand_landmarks:
                hand_frames += 1
            if now - t0 >= RECORD_SECONDS:
                if current != "NONE" and hand_frames < 0.5 * len(frames):
                    message = "Hands not visible - sample discarded, try again"
                else:
                    seq = resample(frames)
                    path = os.path.join(OUT_DIR, current, f"{current}_{counts[current]:03d}.npy")
                    np.save(path, seq)
                    counts[current] += 1
                    message = f"Saved {current} ({counts[current]})"
                state = "idle"


        colour = {"idle": (0, 255, 255), "countdown": (0, 165, 255), "recording": (0, 0, 255)}[state]
        cv2.putText(display, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        if state == "recording":
            cv2.rectangle(display, (0, 0), (display.shape[1] - 1, display.shape[0] - 1), (0, 0, 255), 8)
        y = 65
        for k, name in LABELS.items():
            cv2.putText(display, f"{k}: {name} ({counts[name]})", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25
        cv2.imshow("SAARTHI - record sequences", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if state == "idle" and chr(key) in LABELS:
            current, state, t0 = LABELS[chr(key)], "countdown", time.time()

cap.release()
cv2.destroyAllWindows()