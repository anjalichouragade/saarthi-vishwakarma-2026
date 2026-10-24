# SAARTHI

A portable, offline two-way communication device for Indian Sign Language (ISL) users.
Signs are recognised by camera and shown/spoken as text. Speech is converted to text on screen.

- **Application ID:** VSK26-IITI-AS-0110
- **Competition:** Vishwakarma Awards 2026, Stage 2 (Design Specification)
- **Sub-theme:** Augmentative & Alternative Communication (AAC) Hardware
- **Team:** Anjali Chouragade, Niharika Sharma, Nivedita Kathane, Shivani Mujbaile
- **College:** G H Raisoni College of Engineering, Nagpur

## Folders
- `cad/` enclosure files (.STEP and native source)
- `electronics/` schematic, wiring diagram, BOM
- `software/` Python code (hand tracking, classifier, speech)
- `data/` small sample gesture data
- `docs/` photos, test results, screenshots
- `links.md` external video links

## What is built (Stage 2 prototype, laptop)
- ISL-to-text: both hands and head movement tracked with MediaPipe Holistic, 30-frame sequence features, RandomForest classifier, 6 signs plus an idle class
- Press-to-sign demo with a "Please repeat" response on low confidence
- Live test logs: own setup (48 attempts, 30 correct) and a field test at a school for deaf students (33 attempts, 12 correct)

## Planned (Stage 3)
- Offline speech-to-text for the hearing person's reply
- Continuous (sliding-window) recognition, sign variants, multiple native signers
- Raspberry Pi 4 hardware build

## How to run (Python 3.10-3.12)

```
pip install mediapipe==0.10.14 opencv-python scikit-learn pandas joblib
python software/collect_sequences.py
python software/train_sequences.py
python software/predict_press_to_sign.py
```
