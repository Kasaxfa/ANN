"""Лабораторная работа № 3. Трёхмерные графики, вариант 17.

Запуск из каталога репозитория:
    python lab3_solution.py

Модель использует четыре признака и скрытый слой из трёх нейронов Tanh.
Скрипт сохраняет checkpoint PyTorch, веса в Excel, метрики и интерактивные
3D-графики до/после Tanh и с разделяющей плоскостью.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Daily_Water_Intake.csv"
MODEL_PATH = ROOT / "water_variant17_Tanh_3.pth"
WEIGHTS_PATH = ROOT / "water_variant17_Tanh_3_weights.xlsx"
METRICS_PATH = ROOT / "metrics.json"
HISTORY_PATH = ROOT / "training_metrics.csv"
CORRELATION_PATH = ROOT / "correlation_matrix.csv"
SEED = 42
BATCH_SIZE = 256
EPOCHS = 60
ALL_FEATURES = [
    "Age",
    "Weight (kg)",
    "Daily Water Intake (liters)",
    "Gender",
    "Physical Activity Level",
    "Weather",
]
FEATURES = []
FEATURE_CORRELATIONS: dict[str, float] = {}
GENDER_CODES = {"Female": 0.0, "Male": 1.0}
ACTIVITY_CODES = {"Low": 0.0, "Moderate": 1.0, "High": 2.0}
WEATHER_CODES = {"Cold": 0.0, "Normal": 1.0, "Hot": 2.0}


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class WaterDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 3),
            nn.Tanh(),
            nn.Linear(3, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    global FEATURES, FEATURE_CORRELATIONS

    frame = pd.read_csv(DATASET).dropna().drop_duplicates().reset_index(drop=True)
    frame["target"] = frame["Hydration Level"].map({"Poor": 0, "Good": 1}).astype(int)
    encoded_features = frame[ALL_FEATURES].copy()
    encoded_features["Gender"] = encoded_features["Gender"].map(GENDER_CODES)
    encoded_features["Physical Activity Level"] = encoded_features["Physical Activity Level"].map(ACTIVITY_CODES)
    encoded_features["Weather"] = encoded_features["Weather"].map(WEATHER_CODES)
    targets = frame["target"].to_numpy(dtype=np.float32)

    # Сначала строится корреляционная матрица по всем доступным признакам.
    correlation_frame = encoded_features.copy()
    correlation_frame["Target"] = targets
    correlation = correlation_frame.corr()
    correlation.to_csv(CORRELATION_PATH)
    FEATURE_CORRELATIONS = correlation["Target"].drop("Target").to_dict()

    # Затем выбираются четыре признака с наибольшей абсолютной корреляцией
    # с целевым признаком; только они поступают на вход нейронной сети.
    FEATURES = (
        correlation["Target"].drop("Target").abs().sort_values(ascending=False).head(4).index.tolist()
    )
    features = encoded_features[FEATURES].astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=SEED, stratify=targets
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    return frame, features, x_train, x_test, y_train, y_test, scaler


def evaluate(model: NeuralNetwork, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> dict[str, object]:
    model.eval()
    loss_sum = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for features, targets in loader:
            logits = model(features.to(device)).squeeze(1)
            target_device = targets.to(device)
            loss_sum += loss_fn(logits, target_device).item() * len(targets)
            predictions.extend((torch.sigmoid(logits) >= 0.5).int().cpu().tolist())
            labels.extend(targets.int().tolist())
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "loss": loss_sum / len(loader.dataset),
        "accuracy": float(np.mean(np.array(labels) == np.array(predictions))),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "predictions": predictions,
    }


def save_3d_scene(
    path: Path,
    title: str,
    points: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    plane: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    traces = []
    masks = [
        ((labels == 0) & (predictions == 0), "Класс 0, верно", "red"),
        ((labels == 1) & (predictions == 1), "Класс 1, верно", "green"),
        (labels != predictions, "Ошибка", "purple"),
    ]
    for mask, name, color in masks:
        traces.append(
            go.Scatter3d(
                x=points[mask, 0], y=points[mask, 1], z=points[mask, 2],
                mode="markers", name=name,
                marker={"size": 3.5 if name != "Ошибка" else 5, "color": color, "opacity": 0.7},
            )
        )
    if plane is not None:
        xx, yy, zz = plane
        traces.append(go.Surface(x=xx, y=yy, z=zz, name="Разделяющая плоскость", opacity=0.45, showscale=False, colorscale="Blues"))
    figure = go.Figure(traces)
    figure.update_layout(
        title=title,
        scene={"xaxis_title": "Нейрон 1", "yaxis_title": "Нейрон 2", "zaxis_title": "Нейрон 3"},
        legend={"orientation": "h"}, margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    figure.write_html(path, include_plotlyjs="cdn", full_html=True)


def main() -> None:
    seed_everything()
    frame, encoded_features, x_train, x_test, y_train, y_test, scaler = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(WaterDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    test_loader = DataLoader(WaterDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    model = NeuralNetwork().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history: list[dict[str, float]] = []
    best_state, best_accuracy = None, -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for features, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features.to(device)).squeeze(1)
            loss = loss_fn(logits, targets.to(device))
            loss.backward()
            optimizer.step()
        train_metrics = evaluate(model, train_loader, loss_fn, device)
        test_metrics = evaluate(model, test_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_metrics["loss"], "test_loss": test_metrics["loss"], "train_accuracy": train_metrics["accuracy"], "test_accuracy": test_metrics["accuracy"]})
        if test_metrics["accuracy"] > best_accuracy:
            best_accuracy = test_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)

    all_tensor = torch.tensor(scaler.transform(encoded_features).astype(np.float32), dtype=torch.float32, device=device)
    with torch.no_grad():
        pre_activation = model.network[0](all_tensor).cpu().numpy()
        activations = model.network[1](torch.tensor(pre_activation)).numpy()
        probabilities = torch.sigmoid(model.network[2](torch.tensor(activations))).numpy().ravel()
    labels = frame["target"].to_numpy(dtype=int)
    predictions = (probabilities >= 0.5).astype(int)
    output_layer = model.network[2]
    coefficients = output_layer.weight.detach().cpu().numpy().ravel()
    d = -float(output_layer.bias.detach().cpu().numpy()[0])
    x_min, x_max = activations[:, 0].min(), activations[:, 0].max()
    y_min, y_max = activations[:, 1].min(), activations[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 40), np.linspace(y_min, y_max, 40))
    zz = (d - coefficients[0] * xx - coefficients[1] * yy) / coefficients[2]
    save_3d_scene(ROOT / "water_variant17_before_activation_3_neurons.html", "Вариант 17 — до функции активации", pre_activation, labels, predictions)
    save_3d_scene(ROOT / "water_variant17_tanh_3_neurons.html", "Вариант 17 — Tanh, 3 нейрона", activations, labels, predictions)
    save_3d_scene(ROOT / "water_variant17_tanh_3_neurons_plane_separator.html", "Вариант 17 — Tanh, 3 нейрона, разделяющая плоскость", activations, labels, predictions, (xx, yy, zz))

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(HISTORY_PATH, index=False)
    train_metrics = evaluate(model, train_loader, loss_fn, device)
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    serializable = {key: value for key, value in test_metrics.items() if key not in {"labels", "predictions"}}
    metrics = {
        "variant": 17, "rows": len(frame), "train_rows": len(x_train), "test_rows": len(x_test),
        "candidate_features": ALL_FEATURES,
        "features": FEATURES,
        "feature_correlations_with_target": FEATURE_CORRELATIONS,
        "activation": "Tanh", "hidden_neurons": 3,
        "encoding": {"Hydration Level": {"Poor": 0, "Good": 1}, "Gender": GENDER_CODES, "Physical Activity Level": ACTIVITY_CODES, "Weather": WEATHER_CODES},
        "normalization_mean": scaler.mean_.tolist(), "normalization_std": scaler.scale_.tolist(),
        "plane_coefficients": [float(value) for value in coefficients] + [d],
        "train": {key: value for key, value in train_metrics.items() if key not in {"labels", "predictions"}},
        "test": serializable,
        "history": history,
    }
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with pd.ExcelWriter(WEIGHTS_PATH, engine="openpyxl") as writer:
        pd.DataFrame({"feature": FEATURES, "mean": scaler.mean_, "std": scaler.scale_}).to_excel(writer, sheet_name="Normalization", index=False)
        pd.DataFrame(model.network[0].weight.detach().cpu().numpy(), columns=FEATURES).assign(bias=model.network[0].bias.detach().cpu().numpy()).to_excel(writer, sheet_name="Weights_hidden", index=False)
        pd.DataFrame({"hidden_1": coefficients[0], "hidden_2": coefficients[1], "hidden_3": coefficients[2], "bias": d}, index=[0]).to_excel(writer, sheet_name="Weights_output", index=False)
        pd.DataFrame({"target": labels, "probability": probabilities, "prediction": predictions}).head(2000).to_excel(writer, sheet_name="Predictions", index=False)
    print(f"Test accuracy: {test_metrics['accuracy']:.4%}")
    print(f"Saved model and 3D scenes to {ROOT}")


if __name__ == "__main__":
    main()
