import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from model import SearchCNN
from torch.utils.data import DataLoader, ConcatDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_best_model(best_params, train_dataset, val_dataset, epochs=30, batch_size=128):
    class TempTrial:
        def __init__(self, params): self.params = params
        def suggest_int(self, name, a, b): return self.params[name]
        def suggest_categorical(self, name, choices): return self.params[name]

    best_model = SearchCNN(TempTrial(best_params)).to(device)
    optimizer = optim.Adam(best_model.parameters(), lr=best_params["lr"])
    criterion = nn.CrossEntropyLoss()

    full_train_dataset = ConcatDataset([train_dataset, val_dataset])
    full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        best_model.train()
        running_loss, running_corrects = 0.0, 0
        train_loop = tqdm(full_train_loader, leave=False)
        for xb, yb in train_loop:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = best_model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            running_corrects += (preds.argmax(1) == yb).sum().item()
            train_loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            train_loop.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(full_train_loader.dataset)
        epoch_train_acc = running_corrects / len(full_train_loader.dataset)
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation
        best_model.eval()
        val_running_loss, val_running_corrects = 0.0,0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = best_model(xb)
                loss = criterion(preds, yb)
                val_running_loss += loss.item() * xb.size(0)
                val_running_corrects += (preds.argmax(1) == yb).sum().item()
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_acc = val_running_corrects / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss={epoch_train_loss:.4f} Acc={epoch_train_acc:.4f}, Val Loss={epoch_val_loss:.4f} Acc={epoch_val_acc:.4f}")

    return best_model, train_losses, val_losses, train_accs, val_accs
