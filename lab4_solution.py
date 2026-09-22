#!/usr/bin/env python3
"""Laboratory work 4 / course-report implementation, variant 17.

The experiment encodes three categorical features, trains a small ReLU
classifier and evaluates all 2! * 3! * 3! possible ordinal encodings.  The
script also exports the calculation trace used in the accompanying workbook.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


SEED = 42
FEATURES = ["Age", "Daily Water Intake (liters)", "Gender", "Physical Activity Level", "Weather"]
CATEGORY_FIELDS = ["Gender", "Physical Activity Level", "Weather"]
CATEGORIES = {
    "Gender": ["Female", "Male"],
    "Physical Activity Level": ["Low", "Moderate", "High"],
    "Weather": ["Cold", "Normal", "Hot"],
}
CANONICAL = {
    field: {value: float(index) for index, value in enumerate(values)}
    for field, values in CATEGORIES.items()
}


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


class WaterNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(5, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(torch.relu(self.hidden(x))).squeeze(1)


def encode(data: pd.DataFrame, mapping: dict[str, dict[str, float]]) -> np.ndarray:
    encoded = data[["Age", "Daily Water Intake (liters)"]].astype(float).copy()
    for field in CATEGORY_FIELDS:
        encoded[field] = data[field].map(mapping[field]).astype(float)
    return encoded[FEATURES].to_numpy(dtype=np.float32)


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, indices: np.ndarray) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x[indices])).cpu().numpy()
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities >= 0.5).astype(int)
    target = y[indices]
    matrix = confusion_matrix(target, predictions, labels=[0, 1]).tolist()
    return {
        "loss": float(nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(logits), torch.from_numpy(target.astype(np.float32))
        ).item()),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "confusion_matrix": matrix,
        "predictions": predictions.tolist(),
    }


def error_count(metrics: dict) -> int:
    matrix = metrics["confusion_matrix"]
    return int(matrix[0][1] + matrix[1][0])


def train_model(model: nn.Module, x: np.ndarray, y: np.ndarray,
                train_indices: np.ndarray, test_indices: np.ndarray) -> tuple[list[dict], dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    x_train = torch.from_numpy(x[train_indices])
    y_train = torch.from_numpy(y[train_indices].astype(np.float32))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), batch_size=256, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    history = []
    best_state = deepcopy(model.state_dict())
    best_accuracy = -1.0
    for epoch in range(1, 36):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        train_metrics = evaluate(model, x, y, train_indices)
        test_metrics = evaluate(model, x, y, test_indices)
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "test_loss": test_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
        })
        if test_metrics["accuracy"] > best_accuracy:
            best_accuracy = test_metrics["accuracy"]
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return history, best_state


def all_encodings() -> list[dict[str, dict[str, float]]]:
    result = []
    for gender in itertools.permutations(range(2)):
        for activity in itertools.permutations(range(3)):
            for weather in itertools.permutations(range(3)):
                result.append({
                    "Gender": dict(zip(CATEGORIES["Gender"], map(float, gender))),
                    "Physical Activity Level": dict(zip(CATEGORIES["Physical Activity Level"], map(float, activity))),
                    "Weather": dict(zip(CATEGORIES["Weather"], map(float, weather))),
                })
    return result


def encoding_label(mapping: dict[str, dict[str, float]]) -> str:
    return "; ".join(
        f"{field}: " + ", ".join(f"{name}={int(value)}" for name, value in values.items())
        for field, values in mapping.items()
    )


def evaluate_encoding_search(model: nn.Module, data: pd.DataFrame, y: np.ndarray,
                             scaler: StandardScaler, test_indices: np.ndarray) -> list[dict]:
    rows = []
    for mapping in all_encodings():
        x = scaler.transform(encode(data, mapping)).astype(np.float32)
        metrics = evaluate(model, x, y, test_indices)
        rows.append({
            "mapping": mapping,
            "errors": error_count(metrics),
            "accuracy": metrics["accuracy"],
            "confusion_matrix": metrics["confusion_matrix"],
        })
    return sorted(rows, key=lambda row: (row["errors"], -row["accuracy"]))


def evaluate_interactions(model: nn.Module, data: pd.DataFrame, y: np.ndarray,
                          scaler: StandardScaler, test_indices: np.ndarray,
                          baseline: dict) -> list[dict]:
    baseline_errors = error_count(baseline)
    result = []
    for left_index, right_index in itertools.combinations(range(len(CATEGORY_FIELDS)), 2):
        left, right = CATEGORY_FIELDS[left_index], CATEGORY_FIELDS[right_index]
        best = None
        for left_values in itertools.permutations(range(len(CATEGORIES[left]))):
            for right_values in itertools.permutations(range(len(CATEGORIES[right]))):
                mapping = deepcopy(CANONICAL)
                mapping[left] = dict(zip(CATEGORIES[left], map(float, left_values)))
                mapping[right] = dict(zip(CATEGORIES[right], map(float, right_values)))
                x = scaler.transform(encode(data, mapping)).astype(np.float32)
                metrics = evaluate(model, x, y, test_indices)
                item = {
                    "fields": f"{left} + {right}",
                    "mapping": mapping,
                    "errors": error_count(metrics),
                    "accuracy": metrics["accuracy"],
                    "delta_vs_baseline": baseline_errors - error_count(metrics),
                    "confusion_matrix": metrics["confusion_matrix"],
                }
                if best is None or (item["errors"], -item["accuracy"]) < (best["errors"], -best["accuracy"]):
                    best = item
        result.append(best)
    return result


def save_workbook(path: Path, data: pd.DataFrame, y: np.ndarray, model: WaterNet,
                  baseline_x: np.ndarray, test_indices: np.ndarray, baseline: dict,
                  best: dict, search: list[dict], interactions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_predictions = np.asarray(baseline["predictions"])
    best_x = baseline_x
    # The best mapping is evaluated from the caller and supplied through the
    # temporary attribute below to keep this workbook function self-contained.
    best_predictions = np.asarray(best["predictions"])
    hidden_values = []
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(baseline_x[test_indices[:2000]])
        pre = model.hidden(tensor).numpy()
        hidden = np.maximum(pre, 0.0)
        logits = model.output(torch.from_numpy(hidden)).squeeze(1).numpy()
    for position, row_id in enumerate(test_indices[:2000]):
        hidden_values.append([
            position + 1, int(row_id) + 1, int(y[row_id]), int(base_predictions[position]),
            int(base_predictions[position] != y[row_id]),
            *pre[position].round(8).tolist(), *hidden[position].round(8).tolist(),
            round(float(logits[position]), 8), round(float(1 / (1 + np.exp(-logits[position]))), 8),
        ])
    trace_columns = ["row", "dataset_row", "target", "prediction", "error"]
    trace_columns += [f"s{i}" for i in range(1, 11)] + [f"h{i}" for i in range(1, 11)] + ["logit", "probability"]
    errors = []
    for position, row_id in enumerate(test_indices):
        errors.append([
            position + 1, int(row_id) + 1, int(y[row_id]), int(base_predictions[position]),
            int(base_predictions[position] != y[row_id]), int(best_predictions[position]),
            int(best_predictions[position] != y[row_id]),
        ])
    weight = model.hidden.weight.detach().numpy()
    output_weight = model.output.weight.detach().numpy().ravel()
    raw = data.copy()
    raw.insert(0, "row", np.arange(1, len(raw) + 1))
    input_aug = pd.DataFrame(baseline_x[test_indices[:2000]], columns=[f"x_{f}" for f in FEATURES])
    input_aug.insert(0, "zero_column", 0)
    input_aug.insert(0, "row", np.asarray(test_indices[:2000]) + 1)
    weight_aug = pd.DataFrame(weight.T, columns=[f"h{i}" for i in range(1, 11)])
    weight_aug.insert(0, "unit_column", 0)
    weight_aug.insert(0, "input_row", FEATURES)
    unit = pd.DataFrame([[1] * 11], columns=["unit_column", *[f"h{i}" for i in range(1, 11)]])
    unit.insert(0, "input_row", "unit_row")
    weight_aug = pd.concat([unit, weight_aug], ignore_index=True)
    summary = pd.DataFrame([
        ["variant", 17], ["rows", len(data)], ["train_rows", len(data) - len(test_indices)],
        ["test_rows", len(test_indices)], ["hidden_neurons", 10], ["activation", "ReLU"],
        ["baseline_errors", error_count(baseline)], ["best_errors", best["errors"]],
        ["baseline_accuracy", baseline["accuracy"]], ["best_accuracy", best["accuracy"]],
        ["best_mapping", encoding_label(best["mapping"])],
    ], columns=["parameter", "value"])
    search_table = pd.DataFrame([
        [i + 1, item["errors"], item["accuracy"],
         ",".join(f"{k}={int(v)}" for k, v in item["mapping"]["Gender"].items()),
         ",".join(f"{k}={int(v)}" for k, v in item["mapping"]["Physical Activity Level"].items()),
         ",".join(f"{k}={int(v)}" for k, v in item["mapping"]["Weather"].items())]
        for i, item in enumerate(search)
    ], columns=["rank", "errors", "accuracy", "Gender", "Activity", "Weather"])
    interaction_table = pd.DataFrame([
        [item["fields"], item["errors"], item["accuracy"], item["delta_vs_baseline"], encoding_label(item["mapping"])]
        for item in interactions
    ], columns=["fields", "errors", "accuracy", "delta_vs_baseline", "best_mapping"])
    errors_table = pd.DataFrame(errors, columns=["test_position", "dataset_row", "target", "baseline_prediction", "baseline_error", "best_prediction", "best_error"])
    output_table = pd.DataFrame([[f"hidden_weight_{i + 1}", value] for i, value in enumerate(output_weight)] + [["output_bias", float(model.output.bias.item())]], columns=["parameter", "value"])
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        raw.head(2000).to_excel(writer, sheet_name="Raw_data", index=False)
        input_aug.to_excel(writer, sheet_name="Input_augmented", index=False)
        weight_aug.to_excel(writer, sheet_name="Weights_augmented", index=False)
        pd.DataFrame(hidden_values, columns=trace_columns).to_excel(writer, sheet_name="check_baseline", index=False)
        search_table.to_excel(writer, sheet_name="Encoding_search", index=False)
        interaction_table.to_excel(writer, sheet_name="Interactions", index=False)
        errors_table.to_excel(writer, sheet_name="Error_locations", index=False)
        output_table.to_excel(writer, sheet_name="Output_weights", index=False)


def save_plots(output_dir: Path, search: list[dict], interactions: list[dict]) -> None:
    import matplotlib.pyplot as plt

    assets = output_dir / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.bar(range(1, 21), [row["errors"] for row in search[:20]], color=["#2ca02c"] + ["#4c78a8"] * 19)
    plt.xlabel("Ранг кодировки"); plt.ylabel("Ошибки на тестовой выборке")
    plt.title("ЛР4: ошибки при перестановке кодов категорий")
    plt.tight_layout(); plt.savefig(assets / "encoding_search.png", dpi=160); plt.close()
    plt.figure(figsize=(10, 4))
    values = [row["delta_vs_baseline"] for row in interactions]
    plt.bar([row["fields"] for row in interactions], values, color="#2ca02c")
    plt.ylabel("Снижение числа ошибок"); plt.title("Взаимовлияние категориальных признаков")
    plt.xticks(rotation=15, ha="right"); plt.tight_layout(); plt.savefig(assets / "category_interactions.png", dpi=160); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Daily_Water_Intake.csv")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    seed_everything()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data)
    y = data["Hydration Level"].map({"Poor": 0, "Good": 1}).to_numpy(dtype=np.int64)
    canonical_x = encode(data, CANONICAL)
    all_indices = np.arange(len(data))
    train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(canonical_x[train_indices])
    x = scaler.transform(canonical_x).astype(np.float32)
    model = WaterNet()
    history, state = train_model(model, x, y, train_indices, test_indices)
    baseline = evaluate(model, x, y, test_indices)
    search = evaluate_encoding_search(model, data, y, scaler, test_indices)
    best = search[0]
    best_x = scaler.transform(encode(data, best["mapping"])).astype(np.float32)
    best_metrics = evaluate(model, best_x, y, test_indices)
    best["precision"] = best_metrics["precision"]
    best["recall"] = best_metrics["recall"]
    best["f1"] = best_metrics["f1"]
    best["predictions"] = best_metrics["predictions"]
    interactions = evaluate_interactions(model, data, y, scaler, test_indices, baseline)
    save_workbook(output_dir / "water_variant17_lab4_work.xlsx", data, y, model, x, test_indices, baseline, best, search, interactions)
    torch.save({
        "model_state_dict": state, "architecture": {"input_features": 5, "hidden_neurons": 10, "activation": "ReLU"},
        "features": FEATURES, "canonical_mapping": CANONICAL,
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
    }, output_dir / "water_variant17_ReLU_10_lab4.pth")
    pd.DataFrame(history).to_csv(output_dir / "training_metrics.csv", index=False)
    pd.DataFrame([
        [i + 1, row["errors"], row["accuracy"], encoding_label(row["mapping"]),
         json.dumps(row["confusion_matrix"], ensure_ascii=False)] for i, row in enumerate(search)
    ], columns=["rank", "errors", "accuracy", "mapping", "confusion_matrix"]).to_csv(output_dir / "encoding_search.csv", index=False)
    pd.DataFrame([
        [row["fields"], row["errors"], row["accuracy"], row["delta_vs_baseline"], encoding_label(row["mapping"])]
        for row in interactions
    ], columns=["fields", "errors", "accuracy", "delta_vs_baseline", "mapping"]).to_csv(output_dir / "category_interactions.csv", index=False)
    metrics = {
        "variant": 17, "dataset": "Daily Water Intake & Hydration Patterns Dataset", "rows": len(data),
        "train_rows": len(train_indices), "test_rows": len(test_indices), "features": FEATURES,
        "category_values": CATEGORIES, "canonical_mapping": CANONICAL, "normalization_mean": scaler.mean_.tolist(),
        "normalization_std": scaler.scale_.tolist(), "hidden_neurons": 10, "activation": "ReLU",
        "baseline": {key: value for key, value in baseline.items() if key != "predictions"},
        "best_encoding": {key: value for key, value in best.items() if key != "predictions"},
        "best_metrics": {key: value for key, value in best_metrics.items() if key != "predictions"},
        "interactions": interactions, "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    save_plots(output_dir, search, interactions)
    print(json.dumps({"baseline_errors": error_count(baseline), "best_errors": best["errors"], "best_accuracy": best["accuracy"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
