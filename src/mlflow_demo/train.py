# src/iris_demo/train.py
import argparse
import mlflow
from models import train_xgboost, train_linear, train_nn
from mlflow_demo.models import train_nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xgboost", "linear", "nn"], required=True)
    args = parser.parse_args()

    mlflow.set_experiment("Iris Demo")

    with mlflow.start_run():
        if args.model == "xgboost":
            train_xgboost.run()
        elif args.model == "linear":
            train_linear.run()
        elif args.model == "nn":
            train_nn.run()

if __name__ == "__main__":
    main()
