# ANN

Лабораторная работа № 1 по дисциплине «Искусственные нейронные сети», вариант 17.

## Задание

Используется датасет [Daily Water Intake & Hydration Patterns Dataset](https://www.kaggle.com/datasets/sonalshinde123/daily-water-intake-and-hydration-patterns-dataset). Целевой признак — `Hydration Level`: `Poor = 0`, `Good = 1`.

В работе выполнены очистка данных, удаление дубликатов, one-hot-кодирование категориальных признаков, нормализация, обучение модели с `ReLU`, замена функции активации на `Tanh`, а также сокращение скрытого слоя до 10 нейронов.

## Запуск

```bash
pip install -r requirements.txt
python lab1_solution.py
```

Скрипт должен запускаться из каталога репозитория. После выполнения создаются:

- `daily_water_model_weights.xlsx` — нормализованные тестовые признаки, целевой вектор, веса и смещения итоговой модели `Tanh_10`;
- `training_metrics.csv` — история обучения трёх моделей;
- `metrics.json` — итоговые метрики и матрицы ошибок;
- `confusion_matrices.csv` — данные матриц ошибок.

Полный отчёт находится в файле `Отчёт 1.docx`.
