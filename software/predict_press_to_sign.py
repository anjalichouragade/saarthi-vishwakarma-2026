import csv
import os
import time
from datetime import datetime
import cv2
import joblib
import numpy as np
import mediapipe as mp
from features import extract_features, sequence_features

MODEL_PATH = "software/sign_sequence_model.joblib"
LOG_PATH = "docs/live_test_log.csv"
SEQ_LEN = 30
COUNTDOWN = 1.5        # same as when recording
RECORD_SECONDS = 1.5   # same as when recording
THRESHOLD = 0.45       # below this -> "Please repeat"
INTENDED_KEYS = {"1": "STOP", "2": "WATER", "3": "YES", "4": "NO", "5": "DOCTOR", "6": "HELP"}

model = joblib.load(MODEL_PATH)
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

os.makedirs("docs", exist_ok=True)
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["time", "intended", "predicted", "confidence", "correct"])


def log_attempt(intended, predicted, conf):
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                intended, predicted, f"{conf:.2f}", int(intended == predicted)])


def resample(frames):
    frames = np.array(frames)
    idx = np.linspace(0, len(frames) - 1, SEQ_LEN).round().astype(int)
    return frames[idx]


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera could not be opened.")
    raise SystemExit

IDLE_MSG = "Press SPACE, then sign"
state, t0, frames, hand_frames = "idle", 0.0, [], 0
message, msg_colour, msg_until, details = IDLE_MSG, (0, 255, 255), 0.0, ""
intended = None

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
            message, msg_colour = "GET READY...", (0, 165, 255)
            if now - t0 >= COUNTDOWN:
                state, t0, frames, hand_frames = "recording", now, [], 0
        elif state == "recording":
            message, msg_colour = "SIGN NOW", (0, 0, 255)
            frames.append(extract_features(results))
            if results.left_hand_landmarks or results.right_hand_landmarks:
                hand_frames += 1
            if now - t0 >= RECORD_SECONDS:
                state = "idle"
                if hand_frames < 0.5 * len(frames):
                    message, msg_colour, details = "No hands seen - try again", (0, 165, 255), ""
                    log_attempt(intended, "NO_HANDS", 0.0)
                else:
                    probs = model.predict_proba([sequence_features(resample(frames))])[0]
                    order = probs.argsort()[::-1]
                    best = order[0]
                    label = model.classes_[best]
                    if probs[best] < THRESHOLD or label == "NONE":
                        predicted = "REPEAT"
                        message, msg_colour = "Please repeat", (0, 165, 255)
                    else:
                        predicted = label
                        message, msg_colour = f"{label} ({probs[best]:.0%})", (0, 255, 0)
                    details = " | ".join(f"{model.classes_[i]} {probs[i]:.0%}" for i in order[:3])
                    log_attempt(intended, predicted, probs[best])
                msg_until = now + 4.0
        elif now > msg_until:
            message, msg_colour, details = IDLE_MSG, (0, 255, 255), ""

        if state == "recording":
            cv2.rectangle(display, (0, 0), (display.shape[1] - 1, display.shape[0] - 1), (0, 0, 255), 8)
        cv2.putText(display, message, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, msg_colour, 3)
        cv2.putText(display, details, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"Intended: {intended or '-'}  (keys 1-6)", (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("SAARTHI - press to sign", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if state == "idle":
            if chr(key) in INTENDED_KEYS:
                intended = INTENDED_KEYS[chr(key)]
            elif key == ord(" "):
                if intended is None:
                    message, msg_colour, msg_until = "Press 1-6 to pick the sign first", (0, 165, 255), now + 2
                else:
                    state, t0 = "countdown", time.time()

cap.release()
cv2.destroyAllWindows()