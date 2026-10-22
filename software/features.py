import numpy as np

POSE_IDS = [0, 7, 8, 11, 12]  # nose, left ear, right ear, left shoulder, right shoulder
FEATURE_SIZE = 65 + 65 + 10   # left hand + right hand + head = 140


def _hand_features(hand, centre, scale):
    if hand is None:
        return np.zeros(65)
    pts = np.array([[p.x, p.y, p.z] for p in hand.landmark])
    wrist = pts[0].copy()
    shape = pts - wrist
    s = np.abs(shape[:, :2]).max() or 1.0
    shape = (shape / s).flatten()                        # 63 values: hand shape
    position = (wrist[:2] - centre[:2]) / scale          # 2 values: where the hand is
    return np.concatenate([shape, position])


def extract_features(results):
    if results.pose_landmarks is None:
        return np.zeros(FEATURE_SIZE)
    pose = np.array([[p.x, p.y, p.z] for p in results.pose_landmarks.landmark])
    centre = (pose[11] + pose[12]) / 2
    scale = np.linalg.norm(pose[11][:2] - pose[12][:2]) or 1.0
    head = ((pose[POSE_IDS][:, :2] - centre[:2]) / scale).flatten()   # 10 values
    left = _hand_features(results.left_hand_landmarks, centre, scale)
    right = _hand_features(results.right_hand_landmarks, centre, scale)
    return np.concatenate([left, right, head])
def sequence_features(seq):
    seq = np.asarray(seq)
    return np.concatenate([seq.flatten(), seq.std(axis=0), seq[-1] - seq[0]])