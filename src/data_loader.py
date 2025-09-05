import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load CIFAR-10 Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_val_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
classes = train_val_dataset.classes

# Train/Validation Split
train_len = int(0.9 * len(train_val_dataset))
val_len = len(train_val_dataset) - train_len
train_dataset, val_dataset = random_split(train_val_dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

# Class distribution function
def plot_class_distribution(dataset, title):
    labels = [label for _, label in dataset]
    counts = Counter(labels)
    plt.figure(figsize=(10,5))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette="viridis")
    plt.xticks(ticks=np.arange(len(classes)), labels=classes, rotation=45)
    plt.title(title)
    plt.show()
