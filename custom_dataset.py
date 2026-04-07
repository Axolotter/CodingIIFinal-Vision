import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import datasets, transforms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import requests
import random
import zipfile
import pathlib
from pathlib import Path
from typing import Tuple, Dict, List

from PIL import Image

from timeit import default_timer as timer


data_path = Path("Rebuilt-1/")
image_path = data_path

train_dir = image_path / "train" / "fuel"
test_dir = image_path / "test" 


def split_quarters_pillow(image_path, num):
    img = Image.open(image_path)
    w, h = img.size
    
    # Define the midpoint
    mid_x, mid_y = w // 2, h // 2

    # Define crop boxes: (left, upper, right, lower)
    quarters = [
        img.crop((0, 0, mid_x, mid_y)),        # Top-left
        img.crop((mid_x, 0, w, mid_y)),        # Top-right
        img.crop((0, mid_y, mid_x, h)),        # Bottom-left
        img.crop((mid_x, mid_y, w, h))         # Bottom-right
    ]

    for i, q in enumerate(quarters):
        q.save(f"Rebuilt-1/test-quarters/{num}_quarter_{i+1}.png")



# Set seed
random.seed(42) # <- try changing this and see what happens

# 1. Get all image paths (* means "any combination")
image_path_list = list(image_path.glob("*/*/*.jpg"))

# # 2. Get random image path
# random_image_path = random.choice(image_path_list)

# # 3. Get image class from path name (the image class is the name of the directory where the image is stored)
# image_class = random_image_path.parent.stem

# # 4. Open image
# img = Image.open(random_image_path)

# split_quarters_pillow(random_image_path)

# 5. Print metadata
# print(f"Random image path: {random_image_path}")
# print(f"Image class: {image_class}")
# print(f"Image height: {img.height}")
# print(f"Image width: {img.width}")
# img

# Turning image into array
# img_as_array = np.asarray(img)


def split_all_images(image_paths):
    print("!!!")
    print(image_paths[0])
    for x in range(len(image_paths)):
        image_path = image_paths[x]
        # with Image.open(image) as f:
        #     print(f)
        split_quarters_pillow(image_path, x)
        # print("*")
    print("*")
        

split_all_images(list(test_dir.glob("*.jpg")))

# Plotting
# plt.figure(figsize = (10,7))
# plt.imshow(img_as_array)
# plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
# plt.axis(False)

# plt.savefig('output.png')


data_transform = transforms.Compose([
    transforms.Resize(size = (64, 64)), # Resize our images to a fixed size (such as 64 x 64 here)
    transforms.RandomHorizontalFlip(p = 0.5 ), # Flip the images randomly on the horizontal # p is probability
    transforms.ToTensor() # Turn the image into torch.Tensor
])

# One Image Transform
# print(data_transform(img)) # img should PIL image

def plot_transformed_images(image_paths: list, transform, n = 3, seed = 42):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths.
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k = n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows = 1, ncols = 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize = 16)
            fig.savefig('output.png')

# plot_transformed_images(image_path_list,
#                         transform = data_transform,
#                         n = 3)


# train_data = datasets.ImageFolder(root = train_dir, # target folder of images
#                                   transform = data_transform, # transforms to perform on data (images)
#                                   target_transform = None) # transforms to perform on labels (if necessary)

# test_data = datasets.ImageFolder(root = test_dir,
#                                  transform = data_transform)

# print(f"Train data:\n{train_data}\nTest data:\n{test_data}")

# class_names = train_data.classes
# print(class_names)