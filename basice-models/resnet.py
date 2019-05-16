import torch
import torch.nn as nn


'''  定义    
'''

# for ResNet-18,34
'''
    3x3,64
    3x3,64
    ...
    3x3,128
    3x3,128
'''
class BasicBlock(nn.Module):
    # 放外面
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

     
        # residual 
        # 3x3 + 3x3卷积 不跨层时feature map不变，所以stride=1,channel不变，跨层时feature map减半，stride=2，同时channel扩大两倍
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False), # default padding=0
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        # 1x1卷积,跨层,stride=2,channel变化 不跨层,stride=1
        
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        else:
            self.shortcut = nn.Sequential()   # 当不跨层时，x直接连

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.residual(x) + self.shortcut(x)
        x = self.relu(x)
        return x


# for ResNet-50,101,152
'''
    1x1,64
    3x3,64
    1x1,256
    ...
    1x1,128
    3x3,128
    1x1,512
'''
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
      
        # residual 
        # 1x1 + 3x3 + 1x1卷积
        # in_channels = out_channels, 所以输出是out_channels * 4  
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),  
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * Bottleneck.expansion, kernel_size=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * Bottleneck.expansion)
        )

        # shortcut
        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            )
        else:
            self.shortcut = nn.Sequential()   # 当不跨层时，x直接连
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x): 
        x = self.residual(x) + self.shortcut(x)
        x = self.relu(x)
        return x

  
class ResNet(nn.Module):
    def __init__(self, block_type, layer_list, num_class=100):
        super().__init__()

        # 输入channels
        self.in_channels = 64

        self.prepocess = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(block_type, layer_list[0], 64,  1)
        self.layer2 = self._make_layer(block_type, layer_list[1], 128, 2)    #跨层要feature map减半
        self.layer3 = self._make_layer(block_type, layer_list[2], 256, 2)
        self.layer4 = self._make_layer(block_type, layer_list[3], 512, 2)

        self.avepool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block_type.expansion, num_class)


    def _make_layer(self, block_type, layer_num, out_channels, stride):
        # 合成每一个layer  
        '''
        3x3,64 x2    1x1x64
        3x3,64       3x3x64     x 3
                     1x1x256    64 x 4
        
        3x3,128      1x1x128
        3x3,128      3x3x128
                     1x1x512   128 x 4
        '''
        # 跨层连接 有可能stride=2;形成[2,1,1,1,1,....]
        strides = [stride] + [1] * (layer_num - 1)
        layers = []
        for stride in strides:
            layers.append(block_type(self.in_channels, out_channels,stride))  # 一个residual block完成
            self.in_channels = out_channels * block_type.expansion  # 每次都要保证in_channels是这个值

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.prepocess(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avepool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# 调用
def resnet18():
    """ return ResNet-18 
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return ResNet-34 
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return ResNet-50 
    """
    return ResNet(Bottleneck, [3, 4, 6, 3])

def resnet101():
    """ return ResNet-101 
    """
    return ResNet(Bottleneck, [3, 4, 23, 3])

def resnet152():
    """ return ResNet-152 
    """
    return ResNet(Bottleneck, [3, 8, 36, 3])


if __name__ == "__main__":
    model = resnet50()
    print(model)