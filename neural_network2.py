import torch
import torchvision
import torchvision.transforms as transforms

from custom_dataset import train_dataloader, test_dataloader

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.suptitle(f"Class: {image_path.parent.stem}", fontsize = 16)
    plt.savefig('output2.png')

# get some random training images
dataiter = iter(train_dataloader)
images, labels = next(dataiter)
batch_size = 4
classes = ['1', '2']

# show images
imshow(torchvision.utils.make_grid(images))
# print labels

# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))