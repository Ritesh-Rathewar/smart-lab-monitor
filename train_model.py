"""Train the FORECASTING model.

It looks at a machine's recent history (current CPU/RAM, how fast they are
rising, how jumpy they are) and predicts CPU% and RAM% at 3 points in the future.

Run:  python train_model.py
Outputs: model.pkl, model_info.json (version, accuracy), training_data_sample.csv
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from features import FEATURES, HORIZONS, TARGETS, make_features, status_of
from simulator import MachineSim

LENGTH = 130


def build_dataset(n_sessions=400, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_sessions):
        sim = MachineSim(rng)
        seq = [sim.step() for _ in range(LENGTH)]
        cpu = [s[0] for s in seq]
        ram = [s[1] for s in seq]
        for t in range(6, LENGTH - max(HORIZONS)):
            row = make_features(cpu[:t + 1], ram[:t + 1], seq[t][2], seq[t][3])
            row["session"] = sid
            for h in HORIZONS:
                row[f"cpu_h{h}"] = cpu[t + h]
                row[f"ram_h{h}"] = ram[t + h]
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_dataset()
    df.head(1000).round(2).to_csv("training_data_sample.csv", index=False)

    # Split by machine, so we test on machines the model has never seen
    ids = df["session"].unique()
    train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=1)
    train, test = df[df.session.isin(train_ids)], df[df.session.isin(test_ids)]

    model = RandomForestRegressor(n_estimators=80, max_depth=14,
                                  min_samples_leaf=15, n_jobs=-1, random_state=1)
    model.fit(train[FEATURES], train[TARGETS])
    pred = np.clip(model.predict(test[FEATURES]), 0, 100)

    # Compare with the naive guess "it will stay exactly as it is now"
    n = len(HORIZONS)
    mae_model, mae_naive = [], []
    print(f"{'target':<9}{'model MAE':>10}{'naive MAE':>11}")
    for i, name in enumerate(TARGETS):
        actual = test[name].to_numpy()
        now = test["cpu" if name.startswith("cpu") else "ram"].to_numpy()
        m = float(np.abs(pred[:, i] - actual).mean())
        b = float(np.abs(now - actual).mean())
        mae_model.append(m); mae_naive.append(b)
        print(f"{name:<9}{m:>10.2f}{b:>11.2f}")

    # Does the forecast pick the right status (Normal/Warning/Critical) 12 readings ahead?
    mid = HORIZONS[1]
    j_cpu, j_ram = TARGETS.index(f"cpu_h{mid}"), TARGETS.index(f"ram_h{mid}")
    real = [status_of(c, r, d, nt) for c, r, d, nt in zip(
        test[f"cpu_h{mid}"], test[f"ram_h{mid}"], test["disk"], test["net_mbps"])]
    ours = [status_of(c, r, d, nt) for c, r, d, nt in zip(
        pred[:, j_cpu], pred[:, j_ram], test["disk"], test["net_mbps"])]
    naive = [status_of(c, r, d, nt) for c, r, d, nt in zip(
        test["cpu"], test["ram"], test["disk"], test["net_mbps"])]
    acc = float(np.mean(np.array(real) == np.array(ours)))
    acc_naive = float(np.mean(np.array(real) == np.array(naive)))

    improvement = 1 - np.mean(mae_model) / np.mean(mae_naive)
    now = datetime.now(timezone.utc)
    info = {
        "version": now.strftime("v%Y%m%d-%H%M"),
        "trained_at": now.strftime("%Y-%m-%d %H:%M UTC"),
        "mae_cpu": round(float(np.mean(mae_model[:n])), 2),
        "mae_ram": round(float(np.mean(mae_model[n:])), 2),
        "naive_mae_cpu": round(float(np.mean(mae_naive[:n])), 2),
        "naive_mae_ram": round(float(np.mean(mae_naive[n:])), 2),
        "improvement_pct": round(float(improvement) * 100, 1),
        "status_accuracy": round(acc, 3),
        "naive_status_accuracy": round(acc_naive, 3),
        "samples": int(len(df)),
        "horizons": HORIZONS,
        "features": FEATURES,
    }
    joblib.dump(model, "model.pkl", compress=3)
    with open("model_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("\nStatus accuracy (model):", round(acc, 3), "| naive:", round(acc_naive, 3))
    print("Forecast is", info["improvement_pct"], "% more accurate than the naive guess")
    print("Saved model.pkl and model_info.json ->", info["version"])
