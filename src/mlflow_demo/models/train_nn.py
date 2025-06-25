import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.pytorch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def run():
    # Load and preprocess data
    X, y = load_iris(return_X_y=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Model, loss, optimizer
    model = Net()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(X_train_tensor)
        loss = loss_fn(out, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluation
    with torch.no_grad():
        preds = model(X_test_tensor).argmax(dim=1).numpy()
        acc = accuracy_score(y_test, preds)

    # MLflow logging
    mlflow.log_param("model_type", "pytorch_nn")
    mlflow.log_metric("accuracy", acc)
    mlflow.pytorch.log_model(model, "model")
