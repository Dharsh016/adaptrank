import os
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score


def full_cv_eval(config_path="config.yaml", n_splits=5):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    df           = pd.read_parquet(
        os.path.join(cfg['data']['processed_path'], 'ranking_dataset.parquet')
    )
    feature_cols = cfg['ranker']['feature_cols']

    X      = df[feature_cols].fillna(0).values
    y      = df['relevance'].values
    groups = df['id_student'].values

    params = {
        'objective':        'lambdarank',
        'metric':           'ndcg',
        'ndcg_eval_at':     cfg['ranker']['ndcg_eval_at'],
        'label_gain':       cfg['ranker']['label_gain'],
        'num_leaves':       cfg['ranker'].get('num_leaves', 31),
        'learning_rate':    cfg['ranker'].get('learning_rate', 0.05),
        'min_data_in_leaf': cfg['ranker'].get('min_data_in_leaf', 20),
        'lambda_l2':        cfg['ranker'].get('lambda_l2', 0.1),
        'bagging_fraction': cfg['ranker'].get('bagging_fraction', 0.9),
        'bagging_freq':     cfg['ranker'].get('bagging_freq', 1),
        'verbose':          -1,
    }

    gkf     = GroupKFold(n_splits=n_splits)
    results = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr,  X_val  = X[tr_idx],  X[val_idx]
        y_tr,  y_val  = y[tr_idx],  y[val_idx]
        g_tr  = pd.Series(groups[tr_idx]).value_counts().sort_index().values
        g_val = pd.Series(groups[val_idx]).value_counts().sort_index().values

        model = lgb.train(
            params,
            lgb.Dataset(X_tr, label=y_tr, group=g_tr),
            num_boost_round=400,
            valid_sets=[lgb.Dataset(X_val, label=y_val, group=g_val)],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(-1)]
        )

        preds = model.predict(X_val)
        sids  = groups[val_idx]
        ndcg5_list, ndcg10_list = [], []

        for sid in np.unique(sids):
            mask = sids == sid
            if mask.sum() < 2:
                continue
            tr = y_val[mask].reshape(1, -1)
            pr = preds[mask].reshape(1, -1)
            ndcg5_list.append(ndcg_score(tr, pr, k=5))
            ndcg10_list.append(ndcg_score(tr, pr, k=10))

        fold_result = {
            'fold':       fold + 1,
            'ndcg5':      round(float(np.mean(ndcg5_list)),  4),
            'ndcg10':     round(float(np.mean(ndcg10_list)), 4),
            'n_students': int(len(np.unique(sids)))
        }
        results.append(fold_result)
        print(f"Fold {fold+1}: NDCG@5={fold_result['ndcg5']}  "
              f"NDCG@10={fold_result['ndcg10']}  "
              f"students={fold_result['n_students']}")

    mean5  = round(float(np.mean([r['ndcg5']  for r in results])), 4)
    mean10 = round(float(np.mean([r['ndcg10'] for r in results])), 4)
    std10  = round(float(np.std( [r['ndcg10'] for r in results])), 4)

    print(f"\nMean NDCG@5={mean5}  Mean NDCG@10={mean10} ± {std10}")

    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
    mlflow.set_experiment(cfg['mlflow']['experiment_name'])
    with mlflow.start_run(run_name=f"{n_splits}fold-cv-tuned"):
        mlflow.log_metric("mean_ndcg5",  mean5)
        mlflow.log_metric("mean_ndcg10", mean10)
        mlflow.log_metric("std_ndcg10",  std10)
        for r in results:
            mlflow.log_metric(f"fold{r['fold']}_ndcg10", r['ndcg10'])

    return results, mean5, mean10


if __name__ == "__main__":
    full_cv_eval()