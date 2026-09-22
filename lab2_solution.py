"""Лабораторная работа № 2. Контрастирование нейронной сети, вариант 17."""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Daily_Water_Intake.csv"
WEIGHTS_XLSX = ROOT / "weights_ReLU_20_variant17.xlsx"
METRICS_CSV = ROOT / "contrast_metrics.csv"
HISTORY_CSV = ROOT / "training_metrics_ReLU_20.csv"
SEED, EPOCHS, BATCH_SIZE, HIDDEN = 42, 30, 256, 20
FEATURES = [
    "Age", "Weight (kg)", "Daily Water Intake (liters)",
    "Gender_Female", "Gender_Male", "Physical Activity Level_High",
    "Physical Activity Level_Low", "Physical Activity Level_Moderate",
    "Weather_Cold", "Weather_Hot", "Weather_Normal",
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class ReLU20(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(FEATURES), HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def forward_numpy(x, w0, b0, w1, b1, weight_digits=None, pre_digits=None, activation_digits=None):
    if weight_digits is not None:
        w0, b0 = np.round(w0, weight_digits), np.round(b0, weight_digits)
        w1, b1 = np.round(w1, weight_digits), round(float(b1), weight_digits)
    pre = x @ w0.T + b0
    if pre_digits is not None:
        pre = np.round(pre, pre_digits)
    activation = np.maximum(pre, 0)
    if activation_digits is not None:
        activation = np.round(activation, activation_digits)
    output = activation @ w1 + b1
    probability = sigmoid(output)
    return {
        "pre": pre,
        "activation": activation,
        "output": output,
        "probability": probability,
        "prediction": (probability >= 0.5).astype(int),
    }


def main() -> None:
    seed_everything()
    raw = pd.read_csv(DATASET, encoding="cp1251").dropna().drop_duplicates().reset_index(drop=True)
    y = raw["Hydration Level"].map({"Poor": 0, "Good": 1}).to_numpy(dtype=np.float32)
    x = pd.get_dummies(raw.drop(columns=["Hydration Level"]), dtype=np.float32)
    x = x.reindex(columns=FEATURES, fill_value=0).astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    model = ReLU20().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    history, best_state, best_accuracy = [], None, -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for features, target in loader:
            features, target = features.to(DEVICE), target.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        with torch.no_grad():
            test_prob = torch.sigmoid(model(torch.tensor(x_test, device=DEVICE))).cpu().numpy().ravel()
        test_pred = (test_prob >= 0.5).astype(int)
        accuracy = float((test_pred == y_test).mean())
        history.append({"epoch": epoch, "train_loss": train_loss / len(loader), "accuracy": accuracy})
        if accuracy > best_accuracy:
            best_accuracy, best_state = accuracy, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    state = model.state_dict()
    w0 = state["network.0.weight"].detach().cpu().numpy()
    b0 = state["network.0.bias"].detach().cpu().numpy()
    w1 = state["network.3.weight"].detach().cpu().numpy().ravel()
    b1 = float(state["network.3.bias"].detach().cpu().numpy()[0])

    variants = {
        "baseline": forward_numpy(x_train, w0, b0, w1, b1),
        "roundP_2": forward_numpy(x_train, w0, b0, w1, b1, activation_digits=2),
        "roundP_1": forward_numpy(x_train, w0, b0, w1, b1, activation_digits=1),
        "roundV_2": forward_numpy(x_train, w0, b0, w1, b1, weight_digits=2),
        "roundV_2_P_1": forward_numpy(x_train, w0, b0, w1, b1, weight_digits=2, activation_digits=1),
        "roundPS_2": forward_numpy(x_train, w0, b0, w1, b1, pre_digits=2),
        "roundW_2": forward_numpy(x_train, w0, b0, w1, b1, weight_digits=2),
        "roundW_2_PS_1": forward_numpy(x_train, w0, b0, w1, b1, weight_digits=2, pre_digits=1),
        "roundIn_1": forward_numpy(np.round(x_train, 1), w0, b0, w1, b1),
    }

    summary = []
    for name, result in variants.items():
        cm = confusion_matrix(y_train.astype(int), result["prediction"], labels=[0, 1])
        summary.append({
            "stage": name,
            "correct": int((result["prediction"] == y_train).sum()),
            "errors": int((result["prediction"] != y_train).sum()),
            "accuracy": float((result["prediction"] == y_train).mean()),
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        })
    pd.DataFrame(summary).to_csv(METRICS_CSV, index=False)
    pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)

    inputs = pd.DataFrame(x_train, columns=FEATURES)
    inputs.insert(0, "row", np.arange(1, len(inputs) + 1))
    inputs["target"] = y_train.astype(int)
    exact = variants["baseline"]
    check = pd.concat([
        inputs[["row", "target"] + FEATURES],
        pd.DataFrame(exact["pre"], columns=[f"s{i+1}" for i in range(HIDDEN)]),
        pd.DataFrame(exact["activation"], columns=[f"p{i+1}" for i in range(HIDDEN)]),
        pd.DataFrame({
            "ys": exact["output"],
            "sigmoid": exact["probability"],
            "prediction": exact["prediction"],
            "correct": (exact["prediction"] == y_train).astype(int),
        }),
    ], axis=1)

    with pd.ExcelWriter(WEIGHTS_XLSX, engine="openpyxl") as writer:
        inputs.to_excel(writer, sheet_name="Input", index=False)
        pd.DataFrame(w0, columns=FEATURES).assign(bias=b0).to_excel(writer, sheet_name="Weights", index=False)
        check.to_excel(writer, sheet_name="check", index=False)
        for name, result in variants.items():
            if name == "baseline":
                continue
            frame = pd.DataFrame(result["pre"], columns=[f"s{i+1}" for i in range(HIDDEN)])
            frame["ys"] = result["output"]
            frame["sigmoid"] = result["probability"]
            frame["prediction"] = result["prediction"]
            frame["target"] = y_train.astype(int)
            frame.to_excel(writer, sheet_name=name[:31], index=False)

    print(f"Best test accuracy: {best_accuracy:.4%}")
    print(f"Saved: {WEIGHTS_XLSX}")


if __name__ == "__main__":
    main()
