import torch
import torch.nn as nn
import torch.nn.functional as F



def get_resnet18(num_classes=10):
    return Resnet18(basic_block, num_classes=num_classes)

class basic_block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(basic_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Resnet18(nn.Module):
    def __init__(self, block, num_classes=10):
        super(Resnet18, self).__init__()
        self.in_channel = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 64, 2, stride=1)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):                               # [b,  3, 32, 32]
        out = F.relu(self.bn1(self.conv1(x)))           # [b, 16, 16, 16]
        out = self.layer1(out)                          # [b, 64, 16, 16]
        out = self.layer2(out)                          # [b, 128, 8, 8]
        out = self.layer3(out)                          # [b, 256, 4, 4]
        out = self.layer4(out)                          # [b, 512, 2, 2]
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
