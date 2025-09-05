"""
download_cifar10.py

This script downloads the CIFAR-10 dataset into the `data/` folder.
You can place this script in your GitHub repo and run it to automatically get the dataset.
"""

import torchvision
import torchvision.transforms as transforms
import os

# Make sure the data directory exists
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

# Define transform (optional, just for consistency)
transform = transforms.Compose([
    transforms.ToTensor()
])

# Download CIFAR-10 training and test datasets
print("Downloading CIFAR-10 training dataset...")
train_dataset = torchvision.datasets.CIFAR10(
    root=data_dir, train=True, download=True, transform=transform
)

print("Downloading CIFAR-10 test dataset...")
test_dataset = torchvision.datasets.CIFAR10(
    root=data_dir, train=False, download=True, transform=transform
)

print("CIFAR-10 download completed!")
print(f"Dataset saved in: {data_dir}/cifar-10-batches-py/")
