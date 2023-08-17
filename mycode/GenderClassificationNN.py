#Complex Neural Network
import torch
from torch import nn

class GenderClassificationNN(nn.Module):
    def __init__(self, input_channels=3, image_height = 128, image_width = 128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # every time it passes w and h /2
            nn.Dropout(0.25)
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * (image_height // 16) * (image_width // 16), 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.conv_layers(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        # x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        # print(x.shape)
        return x

#--------------------------------------------------------------------------------------------- 
# Usage
# model = GenderClassificationNN().to(device)

# print(model)