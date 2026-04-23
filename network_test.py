from custom_dataset import data_transform, class_dict
from neural_network import NeuralNetwork, device
from PIL import Image
import torchvision
from torchvision import datasets, transforms
import torch

import torchvision.models as models

from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from pathlib import Path

model = NeuralNetwork().to(device)

model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
# model.eval()

# model = torch.load('model_weights.pth', weights_only=True)



data_path = Path("Rebuilt-1/")
image_path = data_path

train_dir = image_path / "train" / "fuel"
test_dir = image_path / "test" 


qTrain_dir = image_path / "train-quarters"
qTest_dir = image_path / "test-quarters"

# model.eval()

# img = Image.open("Rebuilt-1/test-quarters/fuel/0_quarter_3.png")


def predictImg(img):
    transImage = data_transform(img)
    bImg = torch.unsqueeze(transImage, 0)
    with torch.no_grad():   
        output = model(bImg)
    probabilities = torch.nn.functional.softmax(output, dim=1)

    # Get the predicted class index
    _, predicted_idx = torch.max(output, 1)
    if(predicted_idx == 0):
        print('Predicted: fuel')
        return 0
    if(predicted_idx == 1):
        print('Predicted: no fuel')
        return 1
    # print(f"Predicted class: {predicted_idx.item()}")
    
    # print(f"Probabilities: {probabilities}")

for i in list(qTest_dir.glob("*/*.png")):
    print(i)
    img = Image.open(i)
    predictImg(img)