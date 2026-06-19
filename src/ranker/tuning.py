import os
import yaml
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _cv_ndcg10(params, df, feature_cols, n_splits=3):
    X      = df[feature_cols].fillna(0).values
    y      = df['relevance'].values
    groups = df['id_student'].values

    gkf    = GroupKFold(n_splits=n_splits)
    scores = []

    for tr_idx, val_idx in gkf.split(X, y, groups):
        X_tr,  X_val  = X[tr_idx],  X[val_idx]
        y_tr,  y_val  = y[tr_idx],  y[val_idx]
        g_tr  = pd.Series(groups[tr_idx]).value_counts().sort_index().values
        g_val = pd.Series(groups[val_idx]).value_counts().sort_index().values

        model = lgb.train(
            params,
            lgb.Dataset(X_tr, label=y_tr, group=g_tr),
            num_boost_round=400,
            valid_sets=[lgb.Dataset(X_val, label=y_val, group=g_val)],
            callbacks=[lgb.early_stopping(15), lgb.log_evaluation(-1)]
        )

        preds = model.predict(X_val)
        sids  = groups[val_idx]
        fold_scores = []
        for sid in np.unique(sids):
            mask = sids == sid
            if mask.sum() < 2:
                continue
            fold_scores.append(
                ndcg_score(y_val[mask].reshape(1,-1),
                           preds[mask].reshape(1,-1), k=10)
            )
        if fold_scores:
            scores.append(np.mean(fold_scores))

    return float(np.mean(scores)) if scores else 0.0


def run_tuning(config_path="config.yaml", n_trials=50):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    df           = pd.read_parquet(
        os.path.join(cfg['data']['processed_path'], 'ranking_dataset.parquet')
    )
    feature_cols = cfg['ranker']['feature_cols']
    label_gain   = cfg['ranker']['label_gain']

    def objective(trial):
        params = {
            'objective':         'lambdarank',
            'metric':            'ndcg',
            'ndcg_eval_at':      [5, 10],
            'label_gain':        label_gain,
            'verbose':           -1,
            'num_leaves':        trial.suggest_int('num_leaves', 20, 80),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'min_data_in_leaf':  trial.suggest_int('min_data_in_leaf', 10, 60),
            'lambda_l2':         trial.suggest_float('lambda_l2', 1e-4, 10.0, log=True),
            'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq':      trial.suggest_int('bagging_freq', 1, 5),
        }
        return _cv_ndcg10(params, df, feature_cols)

    study = optuna.create_study(
        direction='maximize',
        study_name='adaptrank-lambdamart',
        storage=None
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_value  = study.best_value
    print(f"\nBest NDCG@10 (3-fold CV): {best_value:.4f}")
    print(f"Best params: {best_params}")

    # Save study
    study_path = os.path.join(cfg['data']['processed_path'], 'optuna_study.pkl')
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)

    # Log to MLflow
    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
    mlflow.set_experiment(cfg['mlflow']['experiment_name'])
    with mlflow.start_run(run_name="optuna-best"):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_ndcg10", best_value)
        mlflow.log_metric("n_trials", n_trials)

    # Update config.yaml with best params
    cfg['ranker'].update(best_params)
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print("config.yaml updated with best params.")

    return best_params, best_value


if __name__ == "__main__":
    run_tuning()