import torch
import torch.nn as nn

class SearchCNN(nn.Module):
    def __init__(self, trial):
        super(SearchCNN, self).__init__()
        layers = []
        in_channels = 3
        n_conv = trial.suggest_int("n_conv_layers", 2, 4)
        for i in range(n_conv):
            out_ch = trial.suggest_categorical(f"n_channels_l{i}", [16, 32, 64])
            kernel = trial.suggest_categorical(f"kernel_size_l{i}", [3,5])
            layers.append(nn.Conv2d(in_channels, out_ch, kernel_size=kernel, padding=kernel//2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_ch
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_channels * (32 // (2 ** n_conv)) ** 2, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
