import torch.nn as nn


class LeNet5(nn.Module):

    def __init__(self) -> None:
        super(LeNet5, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10),
        )

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding='same')
        self.act1 = nn.ReLU()
        self.avg1 = nn.MaxPool2d(kernel_size=2, stride=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding='same')
        self.act2 = nn.ReLU()
        self.avg2 = nn.MaxPool2d(kernel_size=2, stride=3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, padding='same')
        self.act3 = nn.ReLU()
        self.avg3 = nn.MaxPool2d(kernel_size=2, stride=3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.avg1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.avg2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.avg3(x)
        x = x.view(-1, 120)
        return self.classifier(x)
