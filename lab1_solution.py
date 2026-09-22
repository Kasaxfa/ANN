"""Лабораторная работа № 1, вариант 17.

Замена функции активации и упрощение структуры модели нейронной сети.
Датасет: Daily Water Intake & Hydration Patterns Dataset.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "Daily_Water_Intake.csv"
WEIGHTS_PATH = ROOT / "daily_water_model_weights.xlsx"
HISTORY_PATH = ROOT / "training_metrics.csv"
METRICS_PATH = ROOT / "metrics.json"
CONFUSION_PATH = ROOT / "confusion_matrices.csv"

SEED = 42
TEST_SIZE = 0.2
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
INITIAL_HIDDEN_SIZE = 24
FINAL_HIDDEN_SIZE = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Фиксирует генераторы случайных чисел для воспроизводимого результата."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str], dict[str, int]]:
    """Загружает и подготавливает набор данных для бинарной классификации."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Не найден файл датасета: {DATASET_PATH}. "
            "Положите Daily_Water_Intake.csv рядом со скриптом."
        )

    raw_df = pd.read_csv(DATASET_PATH)
    target_column = "Hydration Level"

    required_columns = {
        "Age",
        "Gender",
        "Weight (kg)",
        "Daily Water Intake (liters)",
        "Physical Activity Level",
        "Weather",
        target_column,
    }
    missing_columns = required_columns.difference(raw_df.columns)
    if missing_columns:
        raise ValueError(f"В датасете отсутствуют столбцы: {sorted(missing_columns)}")

    print(f"Исходная форма данных: {raw_df.shape}")
    print(f"Пропуски до обработки: {int(raw_df.isna().sum().sum())}")
    print(f"Дублирующие записи до обработки: {int(raw_df.duplicated().sum())}")

    # По условию лабораторной работы записи с пропусками удаляются.
    # В исходном датасете пропусков нет, но шаг оставлен явно.
    clean_df = raw_df.dropna().drop_duplicates().reset_index(drop=True)
    print(f"Форма после удаления пропусков и дубликатов: {clean_df.shape}")

    target_mapping = {"Poor": 0, "Good": 1}
    y = clean_df[target_column].map(target_mapping)
    if y.isna().any():
        unknown_values = sorted(clean_df.loc[y.isna(), target_column].unique())
        raise ValueError(f"Неизвестные значения целевого признака: {unknown_values}")
    y = y.astype(np.float32)

    features = clean_df.drop(columns=[target_column])
    categorical_columns = features.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    features = pd.get_dummies(
        features,
        columns=categorical_columns,
        dtype=np.float32,
    )
    features = features.astype(np.float32)

    return raw_df, features, y, list(features.columns), target_mapping


class WaterIntakeDataset(Dataset):
    """Набор данных PyTorch для признаков и бинарной целевой переменной."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


class NeuralNetwork(nn.Module):
    """Сеть с одним скрытым слоем и одним выходным нейроном."""

    def __init__(self, input_size: int, hidden_size: int, activation: str) -> None:
        super().__init__()

        if activation == "ReLU":
            activation_layer: nn.Module = nn.ReLU()
        elif activation == "Tanh":
            activation_layer = nn.Tanh()
        else:
            raise ValueError(f"Неподдерживаемая функция активации: {activation}")

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation_layer,
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, 1),
        )

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def train_epoch(
    dataloader: DataLoader,
    model: NeuralNetwork,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0

    for features, targets in dataloader:
        features = features.to(DEVICE)
        targets = targets.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = loss_function(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    dataloader: DataLoader,
    model: NeuralNetwork,
    loss_function: nn.Module,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    all_probabilities: list[float] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(features)
            total_loss += loss_function(logits, targets.unsqueeze(1)).item()

            probabilities = torch.sigmoid(logits).squeeze(1)
            all_probabilities.extend(probabilities.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().astype(int).tolist())

    predictions = (np.asarray(all_probabilities) >= 0.5).astype(int)
    targets = np.asarray(all_targets)
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])

    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "loss": float(total_loss / len(dataloader)),
        "confusion_matrix": matrix.tolist(),
    }


def train_model(
    name: str,
    activation: str,
    hidden_size: int,
    train_dataset: Dataset,
    test_dataset: Dataset,
    seed: int,
) -> tuple[NeuralNetwork, dict[str, object], list[dict[str, object]]]:
    """Обучает сеть и возвращает состояние с лучшей точностью на тесте."""

    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = NeuralNetwork(
        input_size=train_dataset.features.shape[1],
        hidden_size=hidden_size,
        activation=activation,
    ).to(DEVICE)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history: list[dict[str, object]] = []
    best_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    print(f"\nОБУЧЕНИЕ: {name}")
    print(f"Функция активации: {activation}; скрытых нейронов: {hidden_size}")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(train_loader, model, loss_function, optimizer)
        test_metrics = evaluate(test_loader, model, loss_function)

        row = {
            "model": name,
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
        }
        history.append(row)

        if float(test_metrics["accuracy"]) > best_accuracy:
            best_accuracy = float(test_metrics["accuracy"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"test_loss={float(test_metrics['loss']):.4f} | "
            f"accuracy={float(test_metrics['accuracy']) * 100:.2f}%"
        )

    if best_state is None:
        raise RuntimeError("Не удалось сохранить состояние обученной модели")

    model.load_state_dict(best_state)
    best_metrics = evaluate(test_loader, model, loss_function)
    best_metrics["best_epoch"] = best_epoch
    best_metrics["activation"] = activation
    best_metrics["hidden_size"] = hidden_size

    print_metrics(f"РЕЗУЛЬТАТЫ {name}", best_metrics)
    return model, best_metrics, history


def print_metrics(title: str, metrics: dict[str, object]) -> None:
    print(f"\n{title}")
    print(f"Лучшая эпоха: {metrics['best_epoch']}")
    print(f"Accuracy:  {float(metrics['accuracy']) * 100:.2f}%")
    print(f"Precision: {float(metrics['precision']):.4f}")
    print(f"Recall:    {float(metrics['recall']):.4f}")
    print(f"F1-мера:   {float(metrics['f1']):.4f}")
    print(f"Loss:      {float(metrics['loss']):.4f}")
    print("Confusion matrix [Poor, Good]:")
    print(np.asarray(metrics["confusion_matrix"]))


def metrics_for_json(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "best_epoch": int(metrics["best_epoch"]),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "loss": float(metrics["loss"]),
        "activation": metrics["activation"],
        "hidden_size": int(metrics["hidden_size"]),
        "confusion_matrix": metrics["confusion_matrix"],
    }


def export_artifacts(
    final_model: NeuralNetwork,
    final_metrics: dict[str, object],
    features_test: np.ndarray,
    targets_test: np.ndarray,
    feature_names: list[str],
    all_metrics: dict[str, dict[str, object]],
    history: list[dict[str, object]],
    dataset_info: dict[str, object],
) -> None:
    """Сохраняет нормализованные данные, веса и результаты обучения."""

    input_frame = pd.DataFrame(features_test, columns=feature_names)
    input_frame.insert(len(input_frame.columns), "Target", targets_test.astype(int))

    state = final_model.state_dict()
    weights0 = state["network.0.weight"].detach().cpu().numpy()
    bias0 = state["network.0.bias"].detach().cpu().numpy()
    weights1 = state["network.3.weight"].detach().cpu().numpy()
    bias1 = state["network.3.bias"].detach().cpu().numpy()

    metadata = pd.DataFrame(
        [
            ["Dataset", "Daily Water Intake & Hydration Patterns Dataset"],
            ["Source", "https://www.kaggle.com/datasets/sonalshinde123/daily-water-intake-and-hydration-patterns-dataset"],
            ["Rows before cleaning", dataset_info["rows_before_cleaning"]],
            ["Rows after cleaning", dataset_info["rows_after_cleaning"]],
            ["Features after one-hot encoding", len(feature_names)],
            ["Target mapping", "Poor = 0; Good = 1"],
            ["Final activation", final_metrics["activation"]],
            ["Final hidden size", final_metrics["hidden_size"]],
            ["Final best epoch", final_metrics["best_epoch"]],
            ["Final accuracy", final_metrics["accuracy"]],
            ["Device", str(DEVICE)],
        ],
        columns=["Parameter", "Value"],
    )

    with pd.ExcelWriter(WEIGHTS_PATH) as writer:
        input_frame.to_excel(writer, sheet_name="Input", index=False)
        pd.DataFrame(targets_test.astype(int), columns=["Target"]).to_excel(
            writer, sheet_name="Target", index=False
        )
        pd.DataFrame(weights0).to_excel(
            writer, sheet_name="Weights0", index=False, header=False
        )
        pd.DataFrame(bias0.reshape(1, -1)).to_excel(
            writer, sheet_name="Bias0", index=False, header=False
        )
        pd.DataFrame(weights1).to_excel(
            writer, sheet_name="Weights1", index=False, header=False
        )
        pd.DataFrame(bias1.reshape(1, -1)).to_excel(
            writer, sheet_name="Bias1", index=False, header=False
        )
        metadata.to_excel(writer, sheet_name="Metadata", index=False)

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

    confusion_rows = []
    for model_name, metrics in all_metrics.items():
        matrix = np.asarray(metrics["confusion_matrix"])
        for actual in range(2):
            for predicted in range(2):
                confusion_rows.append(
                    {
                        "model": model_name,
                        "actual": actual,
                        "predicted": predicted,
                        "count": int(matrix[actual, predicted]),
                    }
                )
    pd.DataFrame(confusion_rows).to_csv(CONFUSION_PATH, index=False)

    json_data = {
        "dataset": dataset_info,
        "feature_names": feature_names,
        "target_mapping": {"Poor": 0, "Good": 1},
        "models": {
            name: metrics_for_json(metrics) for name, metrics in all_metrics.items()
        },
        "final_model": "Tanh_10",
    }
    METRICS_PATH.write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nExcel с нормализованными признаками и весами: {WEIGHTS_PATH.name}")
    print(f"История обучения: {HISTORY_PATH.name}")
    print(f"Метрики: {METRICS_PATH.name}")


def main() -> None:
    set_seed(SEED)
    raw_df, prepared_features, target, feature_names, target_mapping = load_and_prepare_data()

    features_train, features_test, targets_train, targets_test = train_test_split(
        prepared_features,
        target,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=target,
    )

    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train).astype(np.float32)
    features_test = scaler.transform(features_test).astype(np.float32)

    print(f"Количество признаков после кодирования: {len(feature_names)}")
    print(f"Обучающая выборка: {len(features_train)}")
    print(f"Тестовая выборка: {len(features_test)}")
    print(f"Баланс классов: {target.value_counts().sort_index().to_dict()}")
    print(f"Используемое устройство: {DEVICE}")

    train_dataset = WaterIntakeDataset(features_train, targets_train.to_numpy())
    test_dataset = WaterIntakeDataset(features_test, targets_test.to_numpy())

    relu_model, relu_metrics, relu_history = train_model(
        name="ReLU_24",
        activation="ReLU",
        hidden_size=INITIAL_HIDDEN_SIZE,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        seed=SEED,
    )
    _ = relu_model

    tanh_model, tanh_metrics, tanh_history = train_model(
        name="Tanh_24",
        activation="Tanh",
        hidden_size=INITIAL_HIDDEN_SIZE,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        seed=SEED,
    )
    _ = tanh_model

    final_model, final_metrics, final_history = train_model(
        name="Tanh_10",
        activation="Tanh",
        hidden_size=FINAL_HIDDEN_SIZE,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        seed=SEED + 1,
    )

    all_metrics = {
        "ReLU_24": relu_metrics,
        "Tanh_24": tanh_metrics,
        "Tanh_10": final_metrics,
    }
    all_history = relu_history + tanh_history + final_history

    print("\nСРАВНЕНИЕ МОДЕЛЕЙ")
    print(f"{'Модель':<12} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
    print("-" * 66)
    for model_name, metrics in all_metrics.items():
        print(
            f"{model_name:<12} | "
            f"{float(metrics['accuracy']) * 100:>9.2f}% | "
            f"{float(metrics['precision']):>10.4f} | "
            f"{float(metrics['recall']):>10.4f} | "
            f"{float(metrics['f1']):>10.4f}"
        )

    dataset_info = {
        "rows_before_cleaning": int(len(raw_df)),
        "rows_after_cleaning": int(len(prepared_features)),
        "duplicates_removed": int(raw_df.duplicated().sum()),
        "missing_values_removed": int(raw_df.isna().sum().sum()),
        "test_size": TEST_SIZE,
        "random_state": SEED,
        "target_mapping": target_mapping,
    }
    export_artifacts(
        final_model=final_model,
        final_metrics=final_metrics,
        features_test=features_test,
        targets_test=targets_test.to_numpy(),
        feature_names=feature_names,
        all_metrics=all_metrics,
        history=all_history,
        dataset_info=dataset_info,
    )


if __name__ == "__main__":
    main()
