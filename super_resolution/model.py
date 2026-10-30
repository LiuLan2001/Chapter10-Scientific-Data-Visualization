import math
import torch
import torch.nn as nn

#定义带有正弦激活的全连接层及其对应的初始化
class SineLayer(nn.Module):
    #定义全连接层
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()

        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()
    #对全连接层进行初始化，首层有不同的初始化方式
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0

            self.linear.weight.uniform_(-bound, bound)
    #正弦激活全连接层的前向传播
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

#将正弦激活全连接层组成隐式神经表示
class Siren(nn.Module):
    #在初始化函数中实现隐式神经表示
    def __init__(self, in_features=3, hidden_features=256, hidden_layers=3,
                 out_features=1, first_omega_0=30.0, hidden_omega_0=30.0):
        super().__init__()
        #定义输入层
        layers = [
            SineLayer(in_features, hidden_features,
                      is_first=True, omega_0=first_omega_0)
        ]
        #定义隐藏层
        for _ in range(hidden_layers):
            layers.append(
                SineLayer(hidden_features, hidden_features,
                          omega_0=hidden_omega_0)
            )
        #定义输出层
        final = nn.Linear(hidden_features, out_features)

        bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
        with torch.no_grad():
            final.weight.uniform_(-bound, bound)

        layers.append(final)

        self.net = nn.Sequential(*layers)
    #隐式神经表示的前向传播
    def forward(self, coords):
        return self.net(coords)
