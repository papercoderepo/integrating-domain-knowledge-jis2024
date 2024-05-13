from dataclasses import dataclass

from torch import nn
from torch.utils import data
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNetWrapper(nn.Module):
    def __init__(self, output_dim, weights=ResNet18_Weights.DEFAULT):
        super(ResNetWrapper, self).__init__()
        self.model_ft = models.resnet18(weights=weights)
        self.__adapt_output_layer(output_dim)

    def __adapt_output_layer(self, output_dim):
        num_in_features = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_in_features, output_dim)

    def forward(self, x):
        expanded = x.expand(-1, 3, 64, 216)
        y = self.model_ft(expanded)
        return y


class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 15, 4000)
        self.linear2 = nn.Linear(4000, 20)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        print(input_data.shape)
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # print(x.shape)
        x = self.flatten(x)
        logits = self.linear(x)
        logits = self.linear2(logits)
        return logits
