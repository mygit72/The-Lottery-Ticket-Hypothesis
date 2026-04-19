import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet300(nn.Module):
    """
    LeNet-300-100 architecture for MNIST.
    """
    def __init__(self):
        super(LeNet300, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ConvNet(nn.Module):
    """
    Conv-2, Conv-4, Conv-6 architectures for CIFAR-10.
    """
    def __init__(self, num_conv_layers=2):
        super(ConvNet, self).__init__()
        layers = []
        in_channels = 3
        out_channels = 64
        
        for i in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
            if (i + 1) % 2 == 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                out_channels *= 2
        
        self.features = nn.Sequential(*layers)
        
        # Calculate the size of the flattened features
        # CIFAR-10 is 32x32. MaxPool reduces size by half each time.
        num_pools = num_conv_layers // 2
        final_size = 32 // (2 ** num_pools)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * final_size * final_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
