"""Train the Normal/Warning/Critical classifier on synthetic lab-machine data.
Run:  python train_model.py
Output: model.pkl
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

FEATURES = ["cpu", "ram", "disk", "net_mbps"]
LABELS = {0: "Normal", 1: "Warning", 2: "Critical"}


def make_data(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "cpu": rng.uniform(0, 100, n),
        "ram": rng.uniform(10, 100, n),
        "disk": rng.uniform(10, 100, n),
        "net_mbps": rng.uniform(0, 100, n),
    })

    # Labelling rule (our "ground truth" for the demo).
    # A machine is in trouble when ANY resource is high.
    def label(r):
        if r.cpu > 90 or r.ram > 90 or r.disk > 95:
            return 2  # Critical
        if r.cpu > 70 or r.ram > 75 or r.disk > 85 or r.net_mbps > 85:
            return 1  # Warning
        return 0      # Normal

    df["status"] = df.apply(label, axis=1)
    return df


if __name__ == "__main__":
    df = make_data()
    df.to_csv("training_data.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["status"], test_size=0.2, random_state=1, stratify=df["status"]
    )
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    model.fit(X_train, y_train)

    print(classification_report(y_test, model.predict(X_test),
                                target_names=list(LABELS.values())))
    joblib.dump(model, "model.pkl")
    print("Saved model.pkl")
