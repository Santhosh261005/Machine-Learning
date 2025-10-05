#!/usr/bin/env python3
"""
run_iris_ensemble.py
Small end-to-end script:
 - Load Iris dataset
 - Baseline CV for Bagging, RandomForest, AdaBoost
 - GridSearchCV tuning for each (small grids)
 - Evaluate best estimators on held-out test set
 - Save metrics CSV, plots, and models
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

RANDOM_SEED = 42
OUTDIR = Path("experiments")
OUTDIR.mkdir(exist_ok=True)

def load_data():
    data = load_iris(as_frame=True)
    X = data['data']
    y = data['target']
    feature_names = list(X.columns)
    target_names = list(data['target_names'])
    return X, y, feature_names, target_names

def evaluate_cv(pipe, X, y, cv):
    acc = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=1)
    f1m = cross_val_score(pipe, X, y, cv=cv, scoring='f1_macro', n_jobs=1)
    return {'acc_mean': acc.mean(), 'acc_std': acc.std(), 'f1_mean': f1m.mean(), 'f1_std': f1m.std()}

def main():
    X, y, feat_names, target_names = load_data()
    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # Define pipelines (scaler + classifier)
    pipelines = {
        'bagging': Pipeline([('scaler', StandardScaler()), ('clf',
                     BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM_SEED),
                                       n_jobs=-1, random_state=RANDOM_SEED))]),
        'random_forest': Pipeline([('scaler', StandardScaler()), ('clf',
                     RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1))]),
        'adaboost': Pipeline([('scaler', StandardScaler()), ('clf',
    AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=RANDOM_SEED),
                       random_state=RANDOM_SEED))])
    }

    # 1) Baseline CV
    baseline_results = {}
    print("=== Baseline CV ===")
    for name, pipe in pipelines.items():
        res = evaluate_cv(pipe, X_train, y_train, cv)
        baseline_results[name] = res
        print(f"{name:15} ACC {res['acc_mean']:.4f} ± {res['acc_std']:.4f} | F1_macro {res['f1_mean']:.4f} ± {res['f1_std']:.4f}")

    # 2) Hyperparameter grids (small)
    param_grids = {
        'bagging': {
            'clf__n_estimators': [10, 50, 100],
            'clf__max_samples': [0.6, 0.8, 1.0],
            'clf__max_features': [0.6, 0.8, 1.0]
        },
        'random_forest': {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [None, 3, 5, 10],
            'clf__max_features': ['sqrt', 'log2']
        },
        'adaboost': {
            'clf__n_estimators': [50, 100, 200],
            'clf__learning_rate': [0.01, 0.1, 1.0]
            # base_estimator fixed to depth=1 for simplicity
        }
    }

    best_estimators = {}
    tuning_results = {}

    print("\n=== Grid search (tuning) ===")
    for name, pipe in pipelines.items():
        print(f"\nTuning {name} ...")
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grids[name],
            scoring='f1_macro',
            cv=cv,
            n_jobs=-1,
            verbose=0,
            refit=True
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        best_params = grid.best_params_
        best_score = grid.best_score_
        best_estimators[name] = best
        tuning_results[name] = {'best_score_cv': best_score, 'best_params': best_params}
        print(f"Best CV f1_macro: {best_score:.4f}")
        print("Best params:", best_params)

    # 3) Evaluate best (tuned) models on held-out test set
    print("\n=== Test set evaluation (tuned models) ===")
    test_results = {}
    for name, est in best_estimators.items():
        y_pred = est.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average='macro')
        creport = classification_report(y_test, y_pred, target_names=target_names)
        cm = confusion_matrix(y_test, y_pred)
        test_results[name] = {'accuracy': acc, 'f1_macro': f1m, 'report': creport, 'confusion_matrix': cm}
        print(f"\n{name} -> Test ACC: {acc:.4f}, F1_macro: {f1m:.4f}")
        print("Classification Report:\n", creport)

    # 4) Save metrics & models
    # Create a tidy CSV summarizing baseline CV and tuned CV + test scores
    rows = []
    for name in pipelines.keys():
        row = {
            'model': name,
            'baseline_cv_acc_mean': baseline_results[name]['acc_mean'],
            'baseline_cv_acc_std': baseline_results[name]['acc_std'],
            'baseline_cv_f1_mean': baseline_results[name]['f1_mean'],
            'baseline_cv_f1_std': baseline_results[name]['f1_std'],
            'tuned_cv_f1_mean': tuning_results[name]['best_score_cv'],
            'tuned_params': json.dumps(tuning_results[name]['best_params']),
            'test_accuracy': test_results[name]['accuracy'],
            'test_f1_macro': test_results[name]['f1_macro']
        }
        rows.append(row)
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(OUTDIR / "metrics_summary.csv", index=False)
    print(f"\nSaved metrics to {OUTDIR/'metrics_summary.csv'}")

    # Save models
    for name, est in best_estimators.items():
        fn = OUTDIR / f"best_model_{name}.joblib"
        joblib.dump(est, fn)
        print(f"Saved model {name} -> {fn}")

    # 5) Plots - bar chart comparing baseline vs tuned test scores (accuracy)
    plt.figure(figsize=(8,5))
    names = df_metrics['model']
    baseline_acc = df_metrics['baseline_cv_f1_mean']
    tuned_cv_f1 = df_metrics['tuned_cv_f1_mean']
    test_f1 = df_metrics['test_f1_macro']

    x = np.arange(len(names))
    width = 0.25
    plt.bar(x - width, baseline_acc, width, label='baseline_cv_f1')
    plt.bar(x, tuned_cv_f1, width, label='tuned_cv_f1')
    plt.bar(x + width, test_f1, width, label='test_f1')
    plt.xticks(x, names)
    plt.ylabel('F1 macro')
    plt.title('Baseline CV vs Tuned CV vs Test (F1 macro)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "summary_bar_f1.png", dpi=200)
    print("Saved bar plot ->", OUTDIR / "summary_bar_f1.png")

    # Confusion matrices for each tuned model on test set
    for name, res in test_results.items():
        cm = res['confusion_matrix']
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion matrix: {name}")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(OUTDIR / f"confmat_{name}.png", dpi=200)
        plt.close()
        print(f"Saved confusion matrix -> {OUTDIR/f'confmat_{name}.png'}")

    print("\nAll done. Check the 'experiments' folder for metrics, models, and plots.")

if __name__ == "__main__":
    main()
