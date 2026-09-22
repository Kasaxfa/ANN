"""ЛР4: числовая сеть и возврат категорий через вход с единичными весами."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import differential_evolution
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

SEED = 42
NUMERIC = ['Age', 'Weight (kg)', 'Daily Water Intake (liters)']
CATEGORIES = {'Gender': ['Female', 'Male'], 'Physical Activity Level': ['Low', 'Moderate', 'High'], 'Weather': ['Cold', 'Normal', 'Hot']}
KEYS = [(field, value) for field, values in CATEGORIES.items() for value in values]
MAX_EPOCHS, PATIENCE = 150, 20

class WaterNet(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.hidden = nn.Linear(3, 10)
        self.activation = nn.ReLU() if activation == 'ReLU' else nn.Tanh()
        self.output = nn.Linear(10, 1)
    def forward(self, x):
        return self.output(self.activation(self.hidden(x))).squeeze(1)

def assess(y, logits):
    pred = (logits >= 0).astype(int)
    return {'n': len(y), 'errors': int(np.sum(pred != y)),
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'loss': float(np.mean(np.logaddexp(0, logits) - y * logits)),
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist()}

def forward_numpy(x, w, activation, shift=None):
    z = x @ w['hidden_weight'].T + w['hidden_bias']
    if shift is not None: z = z + shift[:, None]
    h = np.maximum(z, 0) if activation == 'ReLU' else np.tanh(z)
    return z, h, h @ w['output_weight'] + w['output_bias']

def train(x, y, splits, activation):
    torch.manual_seed(SEED)
    model = WaterNet(activation)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    tx, ty = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx[splits['train']], ty[splits['train']]),
        batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    best_loss, bad, best_epoch, history = float('inf'), 0, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss_fn(model(bx), by).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            row = {'activation': activation, 'epoch': epoch}
            for name in ('train', 'validation'):
                idx = splits[name]
                scores = assess(y[idx], model(tx[idx]).numpy())
                row[name+'_loss'], row[name+'_accuracy'] = scores['loss'], scores['accuracy']
        history.append(row)
        if row['validation_loss'] < best_loss - 1e-6:
            best_loss, best_epoch, bad = row['validation_loss'], epoch, 0
            best_state = deepcopy(model.state_dict())
        else: bad += 1
        if bad >= PATIENCE: break
    model.load_state_dict(best_state)
    w = {'hidden_weight': model.hidden.weight.detach().numpy().astype(float),
         'hidden_bias': model.hidden.bias.detach().numpy().astype(float),
         'output_weight': model.output.weight.detach().numpy().ravel().astype(float),
         'output_bias': float(model.output.bias.item())}
    return model, w, history, best_epoch

def optimize_codes(z, design, y, weights, activation, allowed):
    """Коды подбираются только на validation, веса и нормализация фиксированы."""
    columns = [i for i, (field, _) in enumerate(KEYS) if field in allowed]
    evaluations = 0
    def objective(values):
        nonlocal evaluations
        evaluations += 1
        shift = np.sum(design[:, columns] * values, axis=1)
        shifted = z + shift[:, None]
        h = np.maximum(shifted, 0) if activation == 'ReLU' else np.tanh(shifted)
        pred = np.sum(h * weights['output_weight'], axis=1) + weights['output_bias'] >= 0
        return float(np.sum(pred != y) + 1e-6 * np.sum(values ** 2))
    result = differential_evolution(objective, [(-2, 2)] * len(columns), seed=SEED,
        maxiter=45, popsize=6, polish=False, tol=0, atol=0, x0=np.zeros(len(columns)))
    values = np.round(result.x, 3)
    if objective(np.zeros(len(columns))) <= objective(values): values[:] = 0
    for step in (0.1, 0.02, 0.005):
        for _ in range(3):
            old = values.copy()
            for j in range(len(columns)):
                trials = []
                for delta in range(-4, 5):
                    candidate = values.copy()
                    candidate[j] = np.round(np.clip(values[j] + delta * step, -2, 2), 3)
                    trials.append((objective(candidate), candidate))
                values = min(trials, key=lambda item: item[0])[1]
            if np.array_equal(old, values): break
    full = np.zeros(len(KEYS))
    full[columns] = values
    return full, evaluations

def run(data_path, output):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    np.random.seed(SEED)
    raw = pd.read_csv(data_path)
    frame = raw[NUMERIC + list(CATEGORIES) + ['Hydration Level']].copy()
    for field in list(CATEGORIES) + ['Hydration Level']: frame[field] = frame[field].str.strip()
    missing = int(frame.isna().any(axis=1).sum())
    frame = frame.dropna()
    duplicates = int(frame.duplicated().sum())
    frame = frame.loc[~frame.duplicated()].copy()
    frame.insert(0, 'source_csv_row', frame.index + 2)
    frame.reset_index(drop=True, inplace=True)
    for field, values in CATEGORIES.items():
        if not frame[field].isin(values).all(): raise ValueError(f'Unknown category in {field}')
    target = frame['Hydration Level'].map({'Poor': 0, 'Good': 1})
    if target.isna().any(): raise ValueError('Unknown target')
    y = target.to_numpy(dtype=int)
    trainval, test = train_test_split(np.arange(len(frame)), test_size=0.2, stratify=y, random_state=SEED)
    train_idx, val = train_test_split(trainval, test_size=0.25, stratify=y[trainval], random_state=SEED)
    splits = {'train': train_idx, 'validation': val, 'test': test}
    scaler = StandardScaler().fit(frame.loc[train_idx, NUMERIC])
    x = scaler.transform(frame[NUMERIC]).astype(np.float32).astype(float)
    design = np.column_stack([(frame[f] == v).to_numpy(dtype=float) for f,v in KEYS])
    summary = {'variant':17,'seed':SEED,'raw_rows':len(raw),'missing_rows':missing,'duplicates_removed':duplicates,
        'rows':len(frame),'numeric_features':NUMERIC,'categories':CATEGORIES,'category_keys':KEYS,
        'class_counts':np.bincount(y).tolist(),
        'splits':{k:{'n':len(v),'class_counts':np.bincount(y[v]).tolist()} for k,v in splits.items()},
        'normalization_mean':scaler.mean_.tolist(),'normalization_std':scaler.scale_.tolist(),
        'data_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),'models':{}}
    predictions = pd.DataFrame({'source_csv_row':frame.source_csv_row,'target':y,'split':''})
    for name,idx in splits.items(): predictions.loc[idx,'split'] = name
    all_interactions, all_history = [], []
    for activation in ('ReLU','Tanh'):
        print(f'Training {activation} on numeric features only...',flush=True)
        model,w,history,best_epoch = train(x,y,splits,activation)
        z,h,baseline_logits = forward_numpy(x,w,activation)
        with torch.no_grad(): pytorch_logits = model(torch.tensor(x,dtype=torch.float32)).numpy()
        mismatch = int(np.sum((pytorch_logits>=0)!=(baseline_logits>=0)))
        assert mismatch == 0, 'PyTorch / formula predictions differ'
        assert np.array_equal(forward_numpy(x,w,activation,np.zeros(len(x)))[2],baseline_logits)
        weight_hash = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
        best_codes, best_errors = np.zeros(len(KEYS)), assess(y[val],baseline_logits[val])['errors']
        rows = []
        for count in (1,2,3):
            for fields in itertools.combinations(CATEGORIES,count):
                print(f'  {activation}: tuning {" + ".join(fields)} on validation...',flush=True)
                codes,evaluations = optimize_codes(z[val],design[val],y[val],w,activation,fields)
                logits = forward_numpy(x,w,activation,design @ codes)[2]
                scores = {name:assess(y[idx],logits[idx]) for name,idx in splits.items()}
                rows.append({'activation':activation,'fields':list(fields),'codes':codes.tolist(),
                    'evaluations':evaluations,'validation':scores['validation'],'test':scores['test']})
                if scores['validation']['errors'] < best_errors:
                    best_codes,best_errors = codes,scores['validation']['errors']
        tuned_logits = forward_numpy(x,w,activation,design @ best_codes)[2]
        assert weight_hash == hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
        metrics = {'best_epoch':best_epoch,'epochs_run':len(history),'codes':best_codes.tolist(),
            'baseline':{name:assess(y[idx],baseline_logits[idx]) for name,idx in splits.items()},
            'tuned':{name:assess(y[idx],tuned_logits[idx]) for name,idx in splits.items()},
            'all_baseline':assess(y,baseline_logits),'all_tuned':assess(y,tuned_logits),
            'verification':{'pytorch_numpy_prediction_mismatches':mismatch,
                'max_logit_difference':float(np.max(np.abs(pytorch_logits-baseline_logits))),
                'zero_input_preserves_logits':True,'weights_sha256_before_after':weight_hash},
            'weights':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in w.items()}}
        summary['models'][activation] = metrics
        for name,logits in [('baseline',baseline_logits),('tuned',tuned_logits)]:
            predictions[f'{activation}_{name}_logit'] = logits
            predictions[f'{activation}_{name}_pred'] = (logits>=0).astype(int)
            predictions[f'{activation}_{name}_error'] = (logits>=0)!=y
        all_interactions.extend(rows)
        all_history.extend(history)
        torch.save({'model_state_dict':model.state_dict(),'numeric_features':NUMERIC,
            'category_keys':KEYS,'codes':best_codes.tolist(),'scaler_mean':scaler.mean_.tolist(),
            'scaler_scale':scaler.scale_.tolist(),'activation':activation,'hidden_neurons':10},output/f'water_{activation}_lab4.pth')
        print(f"  {activation} test: {metrics['baseline']['test']['errors']} -> {metrics['tuned']['test']['errors']} errors",flush=True)
    summary['interactions'] = all_interactions
    summary['selected_model'] = min(summary['models'],key=lambda a:summary['models'][a]['tuned']['validation']['errors'])
    (output/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    predictions.to_csv(output/'predictions.csv',index=False)
    pd.DataFrame(all_history).to_csv(output/'training_metrics.csv',index=False)
    pd.DataFrame([{'activation':r['activation'],'fields':' + '.join(r['fields']),
        'validation_errors':r['validation']['errors'],'test_errors':r['test']['errors'],
        'codes':json.dumps(r['codes'])} for r in all_interactions]).to_csv(output/'category_interactions.csv',index=False)
    from lab4_workbook import save_workbook
    save_workbook(output/'water_variant17_lab4_work.xlsx',frame,x,y,splits,summary,predictions)
    print('Finished: '+str(output.resolve()),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('Daily_Water_Intake.csv'))
    parser.add_argument('--output-dir',type=Path,default=Path('lab4_reworked'))
    args=parser.parse_args()
    run(args.data,args.output_dir)
