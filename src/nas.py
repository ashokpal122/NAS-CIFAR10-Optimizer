import optuna
from optuna.pruners import HyperbandPruner
from model import SearchCNN
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial, train_loader, val_loader):
    model = SearchCNN(trial).to(device)
    optimizer = optim.Adam(model.parameters(), lr=trial.suggest_loguniform("lr", 1e-4, 1e-2))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):  # short search
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Validation accuracy
        model.eval()
        correct, total = 0,0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        accuracy = correct / total
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return accuracy

def run_nas(train_loader, val_loader, n_trials=10):
    study = optuna.create_study(direction="maximize", pruner=HyperbandPruner())
    study.optimize(lambda trial: objective(trial, train_loader, val_loader), n_trials=n_trials)
    print("Best Trial Params:", study.best_trial.params)
    return study
