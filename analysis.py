import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, recall_score, confusion_matrix, roc_curve, auc

def welch_t_test(df):
    A = df.loc[df['metodo']=='A','puntuacion']
    B = df.loc[df['metodo']=='B','puntuacion']
    t, p = stats.ttest_ind(A, B, equal_var=False)
    return {'mean_A': float(A.mean()), 'mean_B': float(B.mean()), 't_stat': float(t), 'p_value': float(p)}

def bootstrap_mean_diff(df, n_boot=2000, random_state=42):
    rng = np.random.default_rng(random_state)
    A = df.loc[df['metodo']=='A','puntuacion'].values
    B = df.loc[df['metodo']=='B','puntuacion'].values
    diffs = []
    for _ in range(n_boot):
        dA = rng.choice(A, size=A.size, replace=True)
        dB = rng.choice(B, size=B.size, replace=True)
        diffs.append(dA.mean() - dB.mean())
    diffs = np.array(diffs)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return {'diffs': diffs, 'ci_low': float(ci_low), 'ci_high': float(ci_high)}

def fit_linear(df, features=('horas_estudio','horas_suenio','gpa_previo'), alpha=0.0, test_size=0.3, random_state=42):
    X = df[list(features)].values
    y = df['puntuacion'].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)
    if alpha and alpha>0:
        model = Ridge(alpha=alpha)
    else:
        model = LinearRegression()
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)
    return {
        'model': model,
        'MSE': float(mean_squared_error(yte, yhat)),
        'R2': float(r2_score(yte, yhat)),
        'yte': yte,
        'yhat': yhat
    }

def fit_logistic(df, features=('horas_estudio',), C=1.0, threshold=0.5, test_size=0.3, random_state=42):
    X = df[list(features)].values if isinstance(features, (list, tuple)) else df[[features]].values
    if ('metodo' in features) if isinstance(features, (list, tuple)) else (features=='metodo'):
        m = (df['metodo'].values=='B').astype(int).reshape(-1,1)
        if isinstance(features, (list, tuple)) and 'horas_estudio' in features:
            X = np.column_stack([df[['horas_estudio']].values, m])
        else:
            X = m
    y = df['aprueba'].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    log_model = LogisticRegression(C=C, solver='liblinear', max_iter=1000)
    log_model.fit(Xtr, ytr)
    proba = log_model.predict_proba(Xte)[:,1]
    yhat = (proba >= threshold).astype(int)
    acc = accuracy_score(yte, yhat)
    sens = recall_score(yte, yhat, pos_label=1, zero_division=0)
    espec = recall_score(yte, yhat, pos_label=0, zero_division=0)
    fpr, tpr, _ = roc_curve(yte, proba)
    roc_auc = auc(fpr, tpr)
    return {
        'model': log_model,
        'acc': float(acc), 'sens': float(sens), 'espec': float(espec),
        'yte': yte, 'proba': proba, 'yhat': yhat,
        'fpr': fpr, 'tpr': tpr, 'roc_auc': float(roc_auc)
    }
