import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle

df = pd.read_csv("gesture_data.csv")

print("Sample counts per label:")
print(df['label'].value_counts())
print()

X = df.drop("label", axis=1).values
y = df["label"].values

counts = df['label'].value_counts()
min_c  = counts.min()
max_c  = counts.max()
if max_c / min_c > 2:
    print(f"⚠️  Class imbalance detected ({min_c} vs {max_c}). Collect more data for smaller classes.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model...")
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)

scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

labels = sorted(df['label'].unique().tolist())
with open("gesture_labels.pkl", "wb") as f:
    pickle.dump(labels, f)

print(f"\n Model saved -> gesture_model.pkl")
print(f"Labels: {labels}")