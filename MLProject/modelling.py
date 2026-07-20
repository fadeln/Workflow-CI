import os
import sys
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import mlflow
import mlflow.sklearn

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'horseorhumandataset_preprocessing')
TRAIN_DIR = os.path.join(DATA_DIR, 'horse-or-human')
VAL_DIR = os.path.join(DATA_DIR, 'validation-horse-or-human')
IMG_SIZE = (64, 64)

def load_images_and_labels(directory):
    X, y = [], []
    for label, cls_name in enumerate(['horses', 'humans']):
        cls_dir = os.path.join(directory, cls_name)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Directory not found: {cls_dir}")
        for fname in os.listdir(cls_dir):
            img = Image.open(os.path.join(cls_dir, fname)).convert('RGB').resize(IMG_SIZE)
            X.append(np.array(img).flatten().astype(np.float32) / 255.0)
            y.append(label)
    return np.array(X), np.array(y)

mlflow.set_tracking_uri(os.path.join(PROJECT_DIR, "..", "mlruns"))
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Horse_vs_Human_RF"):
    print("Loading training data...")
    X_train, y_train = load_images_and_labels(TRAIN_DIR)
    print(f"Train shape: {X_train.shape}")

    print("Loading validation data...")
    X_val, y_val = load_images_and_labels(VAL_DIR)
    print(f"Val shape: {X_val.shape}")

    print("Training RandomForest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"\nValidation Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['horses', 'humans']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

print("MLflow run selesai.")
