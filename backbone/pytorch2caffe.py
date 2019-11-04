#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import math
import numpy as np

try:
    import caffe
    from caffe import layers as L
    from caffe import params as P
except ImportError:
    pass

def op_name(op_name, m):
    m.op_name = op_name
    return m

class Flatten(nn.Module):
    def __init__(self, axis):
        super(Flatten, self).__init__()
        self.axis = axis

    def forward(self, x):
        assert self.axis == 1
        x = x.reshape(x.shape[0], -1)
        return x

    def generate_caffe_prototxt(self, caffe_net, layer):
        return layer
        layer = L.Flatten(layer, axis=self.axis)
        caffe_net[self.op_name] = layer
        return layer


def flatten(name, axis):
    return op_name(name, Flatten(axis))

def conv_bn_relu(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Sequential(
        op_name(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)),
        op_name(name + '/bn', nn.BatchNorm2d(out_channels)),
        op_name(name + '/relu', nn.ReLU(inplace=True)),
    )

def linear_bn_relu(name, in_features, out_features):
    return nn.Sequential(
        op_name(name + '/FC', nn.Linear(in_features, out_features)),
        op_name(name + '/bn', nn.BatchNorm1d(out_features)),
        op_name(name + '/relu', nn.ReLU(inplace=True)),
    )

def conv_bn(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Sequential(
        op_name(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)),
        op_name(name + '/bn', nn.BatchNorm2d(out_channels)),
    )

def conv(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return op_name(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, True))


def conv_relu(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Sequential(
        op_name(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, True)),
        op_name(name + '/relu', nn.ReLU()),
    )

def conv_prelu(name, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return nn.Sequential(
        op_name(name, nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, True)),
        op_name(name + '/prelu', nn.PReLU()),
    )

def pdp_conv_bn_relu(name, in_channels, out_channels, kernel_size, expand_coef = 6, stride=1):
    return nn.Sequential(
        op_name(name + '/pw1', nn.Conv2d(in_channels, in_channels * expand_coef, kernel_size=1, stride=1, padding=0, bias=False)),
        op_name(name + '/pw1/bn', nn.BatchNorm2d(in_channels * expand_coef)),
        op_name(name + '/pw1/relu', nn.ReLU(inplace=True)),
        
        op_name(name + '/dw', nn.Conv2d(in_channels * expand_coef, in_channels * expand_coef, groups=in_channels * expand_coef, kernel_size=kernel_size, stride=stride, padding=int(((kernel_size - 1) / 2)), bias=False)),
        op_name(name + '/dw/bn', nn.BatchNorm2d(in_channels * expand_coef)),
        op_name(name + '/dw/relu', nn.ReLU(inplace=True)),

        op_name(name + '/pw2', nn.Conv2d(in_channels * expand_coef , out_channels, kernel_size=1, stride=1, padding=0, bias=False)),
        op_name(name + '/pw2/bn', nn.BatchNorm2d(out_channels)),
        op_name(name + '/pw2/relu', nn.ReLU(inplace=True))
    )

def dp_conv_bn_relu(name, in_channels, out_channels, kernel_size, stride=1):
    return nn.Sequential(
        op_name(name + '/dw', nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=kernel_size, stride=stride, padding=int(((kernel_size - 1) / 2)), bias=False)),
        op_name(name + '/dw/bn', nn.BatchNorm2d(in_channels)),
        op_name(name + '/dw/relu', nn.ReLU(inplace=True)),
        op_name(name + '/pw', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)),
        op_name(name + '/pw/bn', nn.BatchNorm2d(out_channels)),
        op_name(name + '/pw/relu', nn.ReLU(inplace=True)),
    )


def d_conv_bn_relu(name, in_channels, kernel_size, stride=1):
    return nn.Sequential(
        op_name(name + '/dw', nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=kernel_size, stride=stride, padding=int(((kernel_size - 1) / 2)), bias=False)),
        op_name(name + '/dw/bn', nn.BatchNorm2d(in_channels)),
        op_name(name + '/dw/relu', nn.ReLU(inplace=True)),
    )

def upsampling(name, in_planes):
    return op_name(name, nn.ConvTranspose2d(in_planes, in_planes, 4, stride=2, padding=1, bias=False, groups=in_planes))


# Define a Basic resnet block
class BasicResnetBlock(nn.Module):
    def __init__(self, name, in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1, direct_plus = False):
        super(BasicResnetBlock, self).__init__()

        self.op_name = name
        
        self.conv_block = nn.Sequential(
                    conv_bn_relu(name = name + '/conv1', in_channels = in_channels, out_channels = out_channels, 
                                               kernel_size = kernel_size, stride = stride, padding = padding),

                    conv_bn_relu(name = name + '/conv2', in_channels = out_channels, out_channels = out_channels, 
                                               kernel_size = 3, stride = 1, padding = 1)
                    )

        if direct_plus and (in_channels == out_channels):
            self.sc_block = None
        else:
            self.sc_block = conv_bn_relu(name = name + '/sc_conv', in_channels = in_channels, out_channels = out_channels, 
                                               kernel_size = 1, stride = stride, padding = 0)

    def forward(self, x):
        if self.sc_block is None:
            out = x + self.conv_block(x)
        else:
            out = self.conv_block(x) + self.sc_block(x)
        return out
    
    def generate_caffe_prototxt(self, caffe_net, layer):
        
        sc_block_out = layer
        conv_block_out = generate_caffe_prototxt(self.conv_block, caffe_net, layer)

        if self.sc_block is not None:
            sc_block_out = generate_caffe_prototxt(self.sc_block, caffe_net, sc_block_out)
        
        sum_out = L.Eltwise(sc_block_out, conv_block_out, operation=P.Eltwise.SUM)
        caffe_net[self.op_name + '/sum'] = sum_out

        return sum_out


class InvertedResidual(nn.Module):
    def __init__(self, name, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.op_name = name
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                op_name(self.op_name + '/1/dw', nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)),
                op_name(self.op_name + '/1/bn', nn.BatchNorm2d(hidden_dim)),
                op_name(self.op_name + '/1/relu', nn.ReLU(inplace=True)),
                # pw-linear
                op_name(self.op_name + '/2/pw-linear', nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                op_name(self.op_name + '/2/bn', nn.BatchNorm2d(oup))
            )
        else:
            self.conv = nn.Sequential(
                # pw
                op_name(self.op_name + '/1/pw', nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)),
                op_name(self.op_name + '/1/bn', nn.BatchNorm2d(hidden_dim)),
                op_name(self.op_name + '/1/relu', nn.ReLU(inplace=True)),
                # dw
                op_name(self.op_name + '/2/dw', nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)),
                op_name(self.op_name + '/2/bn', nn.BatchNorm2d(hidden_dim)),
                op_name(self.op_name + '/2/relu', nn.ReLU(inplace=True)),
                # pw-linear
                op_name(self.op_name + '/3/pw-linear', nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                op_name(self.op_name + '/3/bn', nn.BatchNorm2d(oup))
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

    def generate_caffe_prototxt(self, caffe_net, layer):
        layer_new = generate_caffe_prototxt(self.conv, caffe_net, layer)
        if self.use_res_connect:
            layer_new = L.Eltwise(layer, layer_new, operation=P.Eltwise.SUM)
            caffe_net[self.op_name + '/sum'] = layer_new
        return layer_new

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, name, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.op_name = name

        self.conv1 = conv_bn_relu(name + '/conv1', inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = conv_bn_relu(name + '/conv2', planes, planes, kernel_size=3, stride=1, padding=1)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        ret = out + residual

        return ret

    def generate_caffe_prototxt(self, caffe_net, x):

        residual = x

        out = generate_caffe_prototxt(self.conv1, caffe_net, x)

        out = generate_caffe_prototxt(self.conv2, caffe_net, out)

        if self.downsample is not None:
            residual = generate_caffe_prototxt(self.downsample, caffe_net, x)

        out = L.Eltwise(out, residual, operation=P.Eltwise.SUM)
        caffe_net[self.op_name + '/sum'] = out

        return out

def generate_caffe_prototxt(m, caffe_net, layer):
    if hasattr(m, 'generate_caffe_prototxt'):
        return m.generate_caffe_prototxt(caffe_net, layer)

    if isinstance(m, nn.Sequential):
        for module in m:
            layer = generate_caffe_prototxt(module, caffe_net, layer)
        return layer

    if isinstance(m, nn.Conv2d):
        # if m.bias is None:
        #     param = [dict(lr_mult=1, decay_mult=1)]
        # else:
        #     param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=0)]
        assert m.dilation[0] == m.dilation[1]
        convolution_param = dict(
            num_output=m.out_channels,
            group=m.groups, bias_term=(m.bias is not None),
            # weight_filler=dict(type='msra'),
            dilation=m.dilation[0],
        )
        if m.kernel_size[0] == m.kernel_size[1]:
            convolution_param['kernel_size'] = m.kernel_size[0]
        else:
            convolution_param['kernel_h'] = m.kernel_size[0]
            convolution_param['kernel_w'] = m.kernel_size[1]
        if m.stride[0] == m.stride[1]:
            convolution_param['stride'] = m.stride[0]
        else:
            convolution_param['stride_h'] = m.stride[0]
            convolution_param['stride_w'] = m.stride[1]
        if m.padding[0] == m.padding[1]:
            convolution_param['pad'] = m.padding[0]
        else:
            convolution_param['pad_h'] = m.padding[0]
            convolution_param['pad_w'] = m.padding[1]
        layer = L.Convolution(
            layer,
            # param=param,
            convolution_param=convolution_param,
        )
        caffe_net.tops[m.op_name] = layer
        return layer

    if isinstance(m, nn.ConvTranspose2d):
        # if m.bias is None:
        #     param = [dict(lr_mult=1, decay_mult=1)]
        # else:
        #     param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=0)]
        assert m.dilation[0] == m.dilation[1]
        convolution_param = dict(
            num_output=m.out_channels,
            group=m.groups, bias_term=(m.bias is not None),
            # weight_filler=dict(type='msra'),
            dilation=m.dilation[0],
        )
        if m.kernel_size[0] == m.kernel_size[1]:
            convolution_param['kernel_size'] = m.kernel_size[0]
        else:
            convolution_param['kernel_h'] = m.kernel_size[0]
            convolution_param['kernel_w'] = m.kernel_size[1]
        if m.stride[0] == m.stride[1]:
            convolution_param['stride'] = m.stride[0]
        else:
            convolution_param['stride_h'] = m.stride[0]
            convolution_param['stride_w'] = m.stride[1]
        if m.padding[0] == m.padding[1]:
            convolution_param['pad'] = m.padding[0]
        else:
            convolution_param['pad_h'] = m.padding[0]
            convolution_param['pad_w'] = m.padding[1]
        layer = L.Deconvolution(
            layer,
            # param=param,
            convolution_param=convolution_param,
        )
        caffe_net.tops[m.op_name] = layer
        return layer
    
    if isinstance(m, nn.Linear):
        inner_product_param = dict()
        inner_product_param['num_output'] = m.out_features
        layer = L.InnerProduct(layer, ntop=1, inner_product_param=inner_product_param)
        caffe_net.tops[m.op_name] = layer
        return layer

    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        layer = L.BatchNorm(
            layer, in_place=True,
            batch_norm_param=dict(use_global_stats=True,
                                  eps=m.eps)
            # param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        )
        caffe_net[m.op_name] = layer
        if m.affine:
            layer = L.Scale(
                layer, in_place=True, bias_term=True,
                # filler=dict(type='constant', value=1), bias_filler=dict(type='constant', value=0),
                # param=[dict(lr_mult=1, decay_mult=0), dict(lr_mult=1, decay_mult=0)],
            )
            caffe_net[m.op_name + '/scale'] = layer
        return layer

    if isinstance(m, nn.ReLU):
        layer = L.ReLU(layer, in_place=True)
        caffe_net.tops[m.op_name] = layer
        return layer

    if isinstance(m, nn.PReLU):
        layer = L.PReLU(layer)
        caffe_net.tops[m.op_name] = layer
        return layer

    if isinstance(m, nn.Sigmoid):
        layer = L.Sigmoid(layer, in_place=True)
        caffe_net.tops[m.op_name] = layer
        return layer

    if isinstance(m, nn.Tanh):
        layer = L.TanH(layer, in_place=True)
        caffe_net.tops[m.op_name] = layer
        return layer

    if isinstance(m, nn.AvgPool2d) or isinstance(m, nn.MaxPool2d):
        if isinstance(m, nn.AvgPool2d):
            pooling_param = dict(pool=P.Pooling.AVE)
        else:
            pooling_param = dict(pool=P.Pooling.MAX)
        if isinstance(m.kernel_size, tuple) or isinstance(m.kernel_size, list):
            pooling_param['kernel_h'] = m.kernel_size[0]
            pooling_param['kernel_w'] = m.kernel_size[1]
        else:
            pooling_param['kernel_size'] = m.kernel_size
        if isinstance(m.stride, tuple) or isinstance(m.stride, list):
            pooling_param['stride_h'] = m.stride[0]
            pooling_param['stride_w'] = m.stride[1]
        else:
            pooling_param['stride'] = m.stride
        if isinstance(m.padding, tuple) or isinstance(m.padding, list):
            pooling_param['pad_h'] = m.padding[0]
            pooling_param['pad_w'] = m.padding[1]
        else:
            pooling_param['pad'] = m.padding
        layer = L.Pooling(layer, pooling_param=pooling_param)
        caffe_net.tops[m.op_name] = layer
        return layer
    raise Exception("Unknow module '%s' to generate caffe prototxt." % m)

def convert_weight_from_pytorch_to_caffe(torch_net, caffe_net):
    for name, m in torch_net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            print('convert conv / fullyconnect:', name, m.op_name, m)
            if m.op_name in caffe_net.params:
                caffe_net.params[m.op_name][0].data[...] = m.weight.data.cpu().numpy()
                if m.bias is not None:
                    caffe_net.params[m.op_name][1].data[...] = m.bias.data.cpu().numpy()
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            print('convert bn:', name, m.op_name, m)
            if m.op_name in caffe_net.params:
                caffe_net.params[m.op_name][0].data[...] = m.running_mean.cpu().numpy()
                caffe_net.params[m.op_name][1].data[...] = m.running_var.cpu().numpy()
                caffe_net.params[m.op_name][2].data[...] = 1
                if m.affine:
                    caffe_net.params[m.op_name + '/scale'][0].data[...] = m.weight.data.cpu().numpy()
                    caffe_net.params[m.op_name + '/scale'][1].data[...] = m.bias.data.cpu().numpy()

if __name__ == '__main__':

    class Network(nn.Module):
        def __init__(self, input_size=224, width_mult=1.):
            super(Network, self).__init__()
            self.op_name = 'main'

            input_channel = 32
            last_channel = 64
            interverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 2],
                [4, 24, 2, 2],
                [4, 32, 3, 1],
                [4, 64, 4, 2],
                [4, 96, 3, 1],
            ]

            # building first layer
            assert input_size % 32 == 0
            input_channel = int(input_channel * width_mult)
            self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
            self.features = [conv_bn_relu('init', 3, input_channel, kernel_size=3, stride=2, padding=1)]
            self.features2 = []
            # building inverted residual blocks
            total_stride = 2
            feature1_channels = 0
            stage_index = 0
            for t, c, n, s in interverted_residual_setting:
                output_channel = int(c * width_mult)
                total_stride *= s
                if total_stride <= 8:
                    for i in range(n):
                        if i == 0:
                            self.features.append(InvertedResidual('stage{}_{}'.format(stage_index, i), input_channel, output_channel, s, expand_ratio=t))
                        else:
                            self.features.append(InvertedResidual('stage{}_{}'.format(stage_index, i), input_channel, output_channel, 1, expand_ratio=t))
                        input_channel = output_channel
                        feature1_channels = output_channel
                else:
                    for i in range(n):
                        if i == 0:
                            self.features2.append(InvertedResidual('stage{}_{}'.format(stage_index, i), input_channel, output_channel, s, expand_ratio=t))
                        else:
                            self.features2.append(InvertedResidual('stage{}_{}'.format(stage_index, i), input_channel, output_channel, 1, expand_ratio=t))
                        input_channel = output_channel
                stage_index += 1
            # building last several layers
            self.features1to2 = conv_bn_relu('feature1to2', feature1_channels, self.last_channel, 1, 1, 0)
            self.features2.append(conv_bn_relu('final_stage', input_channel, self.last_channel, 1, 1, 0))
            self.features2.append(upsampling('upscale', self.last_channel))
            # make it nn.Sequential
            self.features = nn.Sequential(*self.features)
            self.features2 = nn.Sequential(*self.features2)

            self.stage1 = self._get_stage_n('feature_stage1', self.last_channel)
            self.stage2 = self._get_stage_n('feature_stage2', self.last_channel + 22)
            self.stage3 = self._get_stage_n('feature_stage3', self.last_channel + 22)

            self.box_regression = []
            self.box_regression.append(dp_conv_bn_relu('box_dpconv1', 64, 32, 3, 2))
            self.box_regression.append(dp_conv_bn_relu('box_dpconv2', 32, 16, 3, 2))
            self.box_regression.append(op_name('box_avg', nn.AvgPool2d(7, 1)))
            self.box_regression.append(conv('box_fc', 16, 4, 1, 1, 0))
            self.box_regression = nn.Sequential(*self.box_regression)

            self._initialize_weights()

        def _get_stage_n(self, name, inplanes):
            layers = []
            layers.append(dp_conv_bn_relu(name + '/2nd/0', inplanes, 48, 3, stride=1))
            for i in range(4):
                layers.append(dp_conv_bn_relu(name + '/2nd/{}'.format(i + 1), 48, 48, 3, stride=1))
            layers.append(conv(name + '/2nd/final', 48, 22, 1, 1, 0))
            layers.append(op_name(name + '2nd/sigmoid', nn.Sigmoid()))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.features(x)
            x_2 = self.features2(x)
            xto2 = self.features1to2(x)

            x = xto2 + x_2

            # stage 1
            stage_1 = self.stage1(x)

            # stage 2
            cat_x_stage1 = torch.cat((x, stage_1), 1)
            stage_2 = self.stage2(cat_x_stage1)

            # stage 3
            cat_x_stage2 = torch.cat((x, stage_2), 1)
            stage_3 = self.stage3(cat_x_stage2)

            # box regression
            box = self.box_regression(x)
            box = box.view((box.shape[0], box.shape[1]))

            return stage_3, box

        def generate_caffe_prototxt(self, caffe_net, layer):
            layer = generate_caffe_prototxt(self.features, caffe_net, layer)
            layer2 = generate_caffe_prototxt(self.features2, caffe_net, layer)
            layer3 = generate_caffe_prototxt(self.features1to2, caffe_net, layer)

            layer_features = L.Eltwise(layer2, layer3, operation=P.Eltwise.SUM)
            caffe_net[self.op_name + '/features'] = layer_features

            layer = generate_caffe_prototxt(self.stage1, caffe_net, layer_features)

            layer_combine1 = L.Concat(layer_features, layer)
            caffe_net[self.op_name + '/layer_combine1'] = layer_combine1

            layer2 = generate_caffe_prototxt(self.stage2, caffe_net, layer_combine1)

            layer_combine2 = L.Concat(layer_features, layer2)
            caffe_net[self.op_name + '/layer_combine2'] = layer_combine2

            layer3 = generate_caffe_prototxt(self.stage3, caffe_net, layer_combine2)

            layer_box = generate_caffe_prototxt(self.box_regression, caffe_net, layer_features)

            return layer3, layer_box

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()


    network = Network(width_mult=0.75)
    print(network)
    caffe_net = caffe.NetSpec()
    layer = L.Input(shape=dict(dim=[1, 3, 224, 224]))
    caffe_net.tops['data'] = layer
    network.generate_caffe_prototxt(caffe_net, layer)
    print(caffe_net.to_proto())
    with open('pytorch2caffe' + '.prototxt', 'w') as f:
        f.write(str(caffe_net.to_proto()))

    caffe_net = caffe.Net('pytorch2caffe' + '.prototxt', caffe.TEST)
    convert_weight_from_pytorch_to_caffe(network, caffe_net)
    caffe_net.save('pytorch2caffe' + '.caffemodel')

    test_input = np.random.rand(1, 3, 224, 224).astype(np.float32) * 10

    caffe_net = caffe.Net('pytorch2caffe.prototxt', 'pytorch2caffe.caffemodel', caffe.TEST)
    caffe_net.blobs['data'].data[...] = test_input
    caffe_output = caffe_net.forward()

    pytorch_input = torch.from_numpy(test_input)
    network.eval()
    with torch.no_grad():
        pytorch_output = network(pytorch_input)
        tmp1, tmp2 = pytorch_output[0].numpy(), pytorch_output[1].numpy()

    print('finish test')

