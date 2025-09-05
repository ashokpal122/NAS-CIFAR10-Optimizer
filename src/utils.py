import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(10,10))
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def show_sample_images(loader, classes):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images, nrow=5))
    print("Sample labels:", [classes[l] for l in labels])
