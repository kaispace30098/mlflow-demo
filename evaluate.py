# evaluate.py
import mlflow

runs = mlflow.search_runs(experiment_names=["Iris Demo"], order_by=["metrics.accuracy DESC"])
best_run = runs.iloc[0]

print("🏆 Best model run ID:", best_run["run_id"])
print("Params:")
for k, v in best_run.items():
    if k.startswith("params."):
        print(f"  {k[7:]} = {v}")

print("Accuracy:", best_run["metrics.accuracy"])
