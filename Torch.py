import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


# 读取MNIST数据集，并使用PyTorch库进行训练和测试。
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()  # 继承 torch.nn.Module 的构造函数

        # 定义第一层全连接层，输入维度为 28 * 28，输出维度为 64
        self.fcl = torch.nn.Linear(28 * 28, 64)

        # 定义第二层全连接层，输入维度为 64，输出维度为 64
        self.fc2 = torch.nn.Linear(64, 64)

        # 定义第三层全连接层，输入维度为 64，输出维度为 64
        self.fc3 = torch.nn.Linear(64, 64)

        # 定义第四层全连接层，输入维度为 64，输出维度为 10
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # 应用 ReLU 激活函数
        x = torch.nn.functional.relu(self.fcl(x))

        # 应用 ReLU 激活函数
        x = torch.nn.functional.relu(self.fc2(x))

        # 应用 ReLU 激活函数
        x = torch.nn.functional.relu(self.fc3(x))

        # 应用 ReLU 激活函数
        x = torch.nn.functional.log_softmax(self.fc4(x),dim=1)

        return x
