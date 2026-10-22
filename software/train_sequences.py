import glob
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from features import sequence_features

X, y = [], []
for folder in glob.glob("data/sequences*"):
    for label in os.listdir(folder):
        for f in glob.glob(os.path.join(folder, label, "*.npy")):
            X.append(sequence_features(np.load(f)))
            y.append(label)

X, y = np.array(X), np.array(y)
for name in sorted(set(y)):
    print(name, (y == name).sum())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(classification_report(y_test, pred))
print(model.classes_)
print(confusion_matrix(y_test, pred, labels=model.classes_))

model.fit(X, y)  # retrain on all data for the final model
joblib.dump(model, "software/sign_sequence_model.joblib")
print("Saved software/sign_sequence_model.joblib")