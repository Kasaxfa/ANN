import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader


# 1. Загрузка данных с правильной обработкой пропусков
file_path = 'kidney_disease.csv'
df = pd.read_csv(file_path, na_values=['\t?', '\t8', '?', ' ?', '  ?', '\t', 'nan', 'NaN', ''])

# Удаляем столбец id, если есть
if 'id' in df.columns:
    df = df.drop(columns=['id'])

print(f"Исходная форма данных: {df.shape}")
print(f"Пропуски до обработки:\n{df.isna().sum().sort_values(ascending=False).head(10)}")

# 2. Обработка целевой переменной
df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})

# 3. Обработка категориальных признаков
categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

# Универсальное кодирование для всех категориальных столбцов
for col in categorical_cols:
    if col in df.columns:
        # Приведение к строке и очистка
        df[col] = df[col].astype(str).str.strip().str.lower()
        
        # Замена специфических значений
        df[col] = df[col].replace({
            'yes, yes': 'yes', ' yes': 'yes', 'no, no': 'no', 
            '\\tno': 'no', '\\tyes': 'yes', ' yes': 'yes',
            ' good': 'good', ' poor': 'poor',
            'false': 'no', 'true': 'yes'
        })
        
        # Кодирование в числа
        df[col] = df[col].map({
            'normal': 0, 'abnormal': 1,
            'good': 0, 'poor': 1,
            'no': 0, 'yes': 1,
            'notpresent': 0, 'present': 1
        })

# 4. Обработка числовых признаков: заполнение пропусков медианой
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if df[col].isna().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# 5. Удаление строк с оставшимися пропусками (если есть нечисловые столбцы)
df = df.dropna().reset_index(drop=True)
df = df.drop_duplicates()

print(f"\nФорма данных после предобработки: {df.shape}")
print(f"Типы данных:\n{df.dtypes.value_counts()}")

# 6. Выделение признаков и целевой переменной
X = df.drop(columns=['classification'])
y = df['classification']

# Проверка на наличие нечисловых значений
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_cols) > 0:
    print(f"\nВнимание: обнаружены нечисловые столбцы: {list(non_numeric_cols)}")
    print("Принудительное преобразование в числовой формат...")
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# 7. Разделение выборок с сохранением баланса классов
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Нормализация
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9. Преобразование в тензоры PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# 10. Создание датасетов
class KidneyDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

train_dataset = KidneyDataset(X_train_tensor, y_train_tensor)
test_dataset = KidneyDataset(X_test_tensor, y_test_tensor)

BATCH_SIZE = 16
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 11. Определение устройства и архитектуры моделей
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_features = X_train.shape[1]

# Модель с функцией активации ReLU
class NeuralNetworkReLU(torch.nn.Module):
    def __init__(self, input_size, hidden_size=24):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size, 1)
        )
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)

# Модель с функцией активации Tanh
class NeuralNetworkTanh(torch.nn.Module):
    def __init__(self, input_size, hidden_size=24):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size, 1)
        )
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)

# 12. Функции обучения и тестирования
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def test_loop(dataloader, model, loss_fn):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y.unsqueeze(1)).item()
            
            pred_prob = torch.sigmoid(pred)
            predicted = (pred_prob > 0.5).float()
            correct += (predicted == y.unsqueeze(1)).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.unsqueeze(1).cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=["Actual 0 (notckd)", "Actual 1 (ckd)"], 
                      columns=["Predicted 0", "Predicted 1"]))
    print(f"Accuracy: {accuracy*100:.1f}%, Loss: {avg_loss:.4f}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}\n")
    
    return accuracy, precision, recall, f1, avg_loss

# 13. Обучение модели с функцией активации ReLU
print("ОБУЧЕНИЕ МОДЕЛИ С ФУНКЦИЕЙ АКТИВАЦИИ ReLU")
print(f"Количество признаков: {n_features}")
print(f"Размер обучающей выборки: {len(train_dataset)}")
print(f"Размер тестовой выборки: {len(test_dataset)}")
print(f"Баланс классов (всего): {y.value_counts().to_dict()}")

model_relu = NeuralNetworkReLU(n_features, hidden_size=24).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_relu.parameters(), lr=0.001)

metrics_history_relu = []
for epoch in range(30):
    train_loss = train_loop(train_dataloader, model_relu, criterion, optimizer)
    print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f}", end=" | ")
    metrics = test_loop(test_dataloader, model_relu, criterion)
    metrics_history_relu.append((epoch+1, *metrics))
    
    # Ранняя остановка при достижении хорошей точности
    if metrics[0] > 0.90 and epoch >= 10:
        print(f"Достигнута точность {metrics[0]*100:.1f}% на эпохе {epoch+1}")
        break

best_epoch_relu = max(metrics_history_relu, key=lambda x: x[1])
print(f"\nРЕЗУЛЬТАТЫ МОДЕЛИ ReLU:")
print(f"Лучшая эпоха: {best_epoch_relu[0]}")
print(f"Точность (Accuracy):    {best_epoch_relu[1]*100:.1f}%")
print(f"Precision:              {best_epoch_relu[2]:.3f}")
print(f"Recall:                 {best_epoch_relu[3]:.3f}")
print(f"F1-мера:                {best_epoch_relu[4]:.3f}\n")

# 14. Обучение модели с функцией активации Tanh
print("ОБУЧЕНИЕ МОДЕЛИ С ФУНКЦИЕЙ АКТИВАЦИИ Tanh")

model_tanh = NeuralNetworkTanh(n_features, hidden_size=24).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_tanh.parameters(), lr=0.001)

metrics_history_tanh = []
for epoch in range(30):
    train_loss = train_loop(train_dataloader, model_tanh, criterion, optimizer)
    print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f}", end=" | ")
    metrics = test_loop(test_dataloader, model_tanh, criterion)
    metrics_history_tanh.append((epoch+1, *metrics))
    
    # Ранняя остановка при достижении хорошей точности
    if metrics[0] > 0.90 and epoch >= 10:
        print(f"Достигнута точность {metrics[0]*100:.1f}% на эпохе {epoch+1}")
        break

best_epoch_tanh = max(metrics_history_tanh, key=lambda x: x[1])
print(f"\nРЕЗУЛЬТАТЫ МОДЕЛИ Tanh:")
print(f"Лучшая эпоха: {best_epoch_tanh[0]}")
print(f"Точность (Accuracy):    {best_epoch_tanh[1]*100:.1f}%")
print(f"Precision:              {best_epoch_tanh[2]:.3f}")
print(f"Recall:                 {best_epoch_tanh[3]:.3f}")
print(f"F1-мера:                {best_epoch_tanh[4]:.3f}\n")

# 15. Сравнение метрик двух моделей
print("СРАВНЕНИЕ МОДЕЛЕЙ ReLU vs Tanh")
print(f"{'Метрика':<15} | {'ReLU':>10} | {'Tanh':>10} | {'Разница':>10}")
print(f"{'Accuracy':<15} | {best_epoch_relu[1]*100:>9.1f}% | {best_epoch_tanh[1]*100:>9.1f}% | {abs(best_epoch_relu[1]-best_epoch_tanh[1])*100:>9.1f}%")
print(f"{'Precision':<15} | {best_epoch_relu[2]:>10.3f} | {best_epoch_tanh[2]:>10.3f} | {abs(best_epoch_relu[2]-best_epoch_tanh[2]):>10.3f}")
print(f"{'Recall':<15} | {best_epoch_relu[3]:>10.3f} | {best_epoch_tanh[3]:>10.3f} | {abs(best_epoch_relu[3]-best_epoch_tanh[3]):>10.3f}")
print(f"{'F1-мера':<15} | {best_epoch_relu[4]:>10.3f} | {best_epoch_tanh[4]:>10.3f} | {abs(best_epoch_relu[4]-best_epoch_tanh[4]):>10.3f}")

# 16. Сохранение весов моделей в Excel
weights0_relu = model_relu.state_dict()['network.0.weight'].cpu().numpy()
bias0_relu = model_relu.state_dict()['network.0.bias'].cpu().numpy()
weights1_relu = model_relu.state_dict()['network.3.weight'].cpu().numpy()
bias1_relu = model_relu.state_dict()['network.3.bias'].cpu().numpy()

weights0_tanh = model_tanh.state_dict()['network.0.weight'].cpu().numpy()
bias0_tanh = model_tanh.state_dict()['network.0.bias'].cpu().numpy()
weights1_tanh = model_tanh.state_dict()['network.3.weight'].cpu().numpy()
bias1_tanh = model_tanh.state_dict()['network.3.bias'].cpu().numpy()

with pd.ExcelWriter('kidney_model_weights.xlsx') as writer:
    df_input = pd.DataFrame(X_test, columns=[f'f{i}' for i in range(n_features)])
    df_input['target'] = y_test.values
    df_input.to_excel(writer, sheet_name='Input', index=False)
    
    # Веса и смещения модели ReLU
    pd.DataFrame(weights0_relu).to_excel(writer, sheet_name='Weights0_ReLU', index=False, header=False)
    pd.DataFrame(bias0_relu.reshape(1, -1)).to_excel(writer, sheet_name='Bias0_ReLU', index=False, header=False)
    pd.DataFrame(weights1_relu).to_excel(writer, sheet_name='Weights1_ReLU', index=False, header=False)
    pd.DataFrame(bias1_relu.reshape(1, -1)).to_excel(writer, sheet_name='Bias1_ReLU', index=False, header=False)
    
    # Веса и смещения модели Tanh
    pd.DataFrame(weights0_tanh).to_excel(writer, sheet_name='Weights0_Tanh', index=False, header=False)
    pd.DataFrame(bias0_tanh.reshape(1, -1)).to_excel(writer, sheet_name='Bias0_Tanh', index=False, header=False)
    pd.DataFrame(weights1_tanh).to_excel(writer, sheet_name='Weights1_Tanh', index=False, header=False)
    pd.DataFrame(bias1_tanh.reshape(1, -1)).to_excel(writer, sheet_name='Bias1_Tanh', index=False, header=False)

print("\n✅ Веса моделей и ВСЯ тестовая выборка сохранены в файл: kidney_model_weights.xlsx")