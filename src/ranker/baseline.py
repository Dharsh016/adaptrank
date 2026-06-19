import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import yaml
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import ndcg_score


def train_baseline(config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_parquet(
        os.path.join(cfg['data']['processed_path'], 'ranking_dataset.parquet')
    )
    feature_cols = cfg['ranker']['feature_cols']

    X      = df[feature_cols].fillna(0).values
    y      = df['relevance'].values
    groups = df['id_student'].values

    # Group-aware split: no student appears in both train and test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # LambdaMART needs group sizes in the same order as rows
    g_train = (
        pd.Series(groups[train_idx])
        .value_counts()
        .sort_index()
        .values
    )
    g_test = (
        pd.Series(groups[test_idx])
        .value_counts()
        .sort_index()
        .values
    )

    train_set = lgb.Dataset(X_train, label=y_train, group=g_train)
    test_set  = lgb.Dataset(X_test,  label=y_test,  group=g_test,
                            reference=train_set)

    params = {
        'objective':        cfg['ranker']['objective'],
        'metric':           cfg['ranker']['metric'],
        'ndcg_eval_at':     cfg['ranker']['ndcg_eval_at'],
        'num_leaves':       cfg['ranker']['num_leaves'],
        'learning_rate':    cfg['ranker']['learning_rate'],
        'min_data_in_leaf': cfg['ranker']['min_data_in_leaf'],
        'lambda_l2':        cfg['ranker']['lambda_l2'],
        'bagging_fraction': cfg['ranker']['bagging_fraction'],
        'bagging_freq':     cfg['ranker']['bagging_freq'],
        'label_gain':       cfg['ranker']['label_gain'],
        'verbose':          -1,
    }

    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
    mlflow.set_experiment(cfg['mlflow']['experiment_name'])

    with mlflow.start_run(run_name="baseline-lambdamart"):
        model = lgb.train(
            params,
            train_set,
            num_boost_round=cfg['ranker']['n_estimators'],
            valid_sets=[test_set],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)]
        )

        preds       = model.predict(X_test)
        student_ids = groups[test_idx]

        ndcg5_list, ndcg10_list = [], []
        for sid in np.unique(student_ids):
            mask = student_ids == sid
            if mask.sum() < 2:
                continue
            true_rel    = y_test[mask].reshape(1, -1)
            pred_scores = preds[mask].reshape(1, -1)
            ndcg5_list.append(ndcg_score(true_rel, pred_scores, k=5))
            ndcg10_list.append(ndcg_score(true_rel, pred_scores, k=10))

        mean_ndcg5  = float(np.mean(ndcg5_list))
        mean_ndcg10 = float(np.mean(ndcg10_list))

        mlflow.log_params(params)
        mlflow.log_metric("ndcg_at_5",  mean_ndcg5)
        mlflow.log_metric("ndcg_at_10", mean_ndcg10)
        mlflow.lightgbm.log_model(model, "baseline_model")

        print(f"\nBaseline NDCG@5:  {mean_ndcg5:.4f}")
        print(f"Baseline NDCG@10: {mean_ndcg10:.4f}")

    model.save_model(
        os.path.join(cfg['data']['processed_path'], 'baseline_model.txt')
    )
    return model, mean_ndcg5, mean_ndcg10


if __name__ == "__main__":
    train_baseline()