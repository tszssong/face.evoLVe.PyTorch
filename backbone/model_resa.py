import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from .basic_layers import ResidualBlock
from .attention_module import AttentionModule_stage1, AttentionModule_stage2, AttentionModule_stage3, AttentionModule_stage0
from torchstat import stat
# from basic_layers import ResidualBlock
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, ReLU, Dropout, MaxPool2d
from torch.nn import AdaptiveAvgPool2d, Sequential, Module, PReLU, Sigmoid
class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d( channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = PReLU()
        self.fc2 = Conv2d( channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x

class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride=1):
        super(bottleneck_IR_SE, self).__init__()
        self.downsample = None
        if stride > 1 or in_channel!=depth:
            self.downsample = Sequential(
                Conv2d(in_channel, depth, (1,1), stride = stride, bias = False),
                BatchNorm2d(depth),
            )
       
        self.res_layer = Sequential(
            # BatchNorm2d(in_channel),
            # Conv2d(in_channel, depth, (1, 1), bias=False),
            # PReLU(depth),
            # BatchNorm2d(depth),
            # Conv2d(depth, depth, (3, 3), padding = 1, bias=False),
            # PReLU(depth),
            # Conv2d(depth, depth, (1, 1), stride, bias=False),
            # BatchNorm2d(depth),
            # SEModule(depth, 16)
            # BatchNorm2d(in_channel),
            # Conv2d(in_channel, in_channel, (1, 1), bias=False),
            # PReLU(in_channel),
            # BatchNorm2d(in_channel),
            # Conv2d(in_channel, in_channel, (3, 3), padding = 1, bias=False),
            # PReLU(in_channel),
            # Conv2d(in_channel, depth, (1, 1), stride, bias=False),
            # BatchNorm2d(depth),
            # SEModule(depth, 16)
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        res = self.res_layer(x)
        return res + identity

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class RAModel_92(Module):

    def __init__(self, input_size):
        super(RAModel_92, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True)
            nn.PReLU(64)
        )
        # self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        self.residual_block1 = bottleneck_IR_SE(64, 64, 2)                 #56x56
        self.attention_module1 = AttentionModule_stage1(64, 64)
        self.attention_module1_2 = AttentionModule_stage1(64, 64)
        self.attention_module1_3 = AttentionModule_stage1(64, 64)

        self.residual_block2 = bottleneck_IR_SE(64, 128, 2)                #28x28
        self.attention_module2 = AttentionModule_stage2(128, 128)
        self.attention_module2_2 = AttentionModule_stage2(128, 128)
        self.attention_module2_3 = AttentionModule_stage2(128, 128)     
        self.attention_module2_4 = AttentionModule_stage2(128, 128)     
        self.attention_module2_5 = AttentionModule_stage2(128, 128)     
        self.attention_module2_6 = AttentionModule_stage2(128, 128)          

        self.residual_block3 = bottleneck_IR_SE(128, 256, 2)               #14x14
        self.attention_module3 = AttentionModule_stage3(256, 256)
        self.attention_module3_2 = AttentionModule_stage3(256, 256)

        self.residual_block4 = bottleneck_IR_SE(256, 512, 2)              #7x7
        self.residual_block5 = bottleneck_IR_SE(512, 512)
        self.residual_block6 = bottleneck_IR_SE(512, 512)
        
        self.output_layer = nn.Sequential( nn.BatchNorm2d(512),
                                           nn.Dropout(),
                                           Flatten(),
                                           nn.Linear(512 * 7 * 7, 512),
                                           nn.BatchNorm1d(512))

    def forward(self, x):
        out = self.conv1(x)
        # out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.attention_module1_2(out)
        out = self.attention_module1_3(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.attention_module2_3(out)
        out = self.attention_module2_4(out)
        out = self.attention_module2_5(out)
        out = self.attention_module2_6(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.output_layer(out)
        # out = self.mpool2(out)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)

        return out

def RA_92(input_size, **kwargs):
    model = RAModel_92(input_size)
    return model
if __name__ == '__main__':
    model=RA_92(input_size=[112,112])
    stat(model, (3,112,112))