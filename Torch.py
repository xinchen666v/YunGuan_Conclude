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
        x = torch.nn.functional.log_softmax(self.fc4(x),dim=1)  # 为了提高运算的稳定性，套上了一个对数运算

        return x

# 定义数据加载器
def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])  # 定义张量，也就是多维数组
    data_set = MNIST("", is_train, transform=to_tensor, download=True)  # 下载数据集，空表示当前目录，to_tensor表示训练集
    return DataLoader(data_set, batch_size=15, shuffle=True)  # shuffle=True表示打乱数据集


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:   # 从测试集中批次取出数据
            outputs = net.forward(x.view(-1, 28 * 28))  # 计算神经网络的预测值
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:  # argmax函数计算一个数列中的最大值的序号
                    n_correct += 1
                n_total += 1        # 累加正确预测的数量
    return n_correct / n_total      # 返回正确率


def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()   # 初始化神经网络

    print("initial accuracy:", evaluate(test_data, net))    # 打印初始正确率，理论上接近0.1

    # pytorch的固定写法
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(2):      # 每个轮次，就是一个epoch，这样可以提高数据集的利用率
        for (x, y) in train_data:
            net.zero_grad()  # 初始化
            output = net.forward(x.view(-1, 28 * 28))   # 正向传播
            loss = torch.nn.functional.nll_loss(output, y)      # 计算差值，nll_loss是对数损失函数，是为了匹配上面的log_softmax
            loss.backward()     # 反向误差传播
            optimizer.step()        # 优化网络参数
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    # 随机抽取三张图片
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction:" + str(predict))
    plt.show()


if __name__ == "__main__":
    main()
