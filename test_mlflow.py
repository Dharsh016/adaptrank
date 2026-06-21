#!/usr/bin/env python
import sys
sys.path.insert(0,'.')
import mlflow
import yaml

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
mlflow.set_experiment(cfg['mlflow']['experiment_name'])

with mlflow.start_run(run_name='connection-test-sqlite'):
    mlflow.log_metric('test', 1.0)

print('✓ Success! MLflow is now using SQLite backend.')
