import sys
sys.path.insert(0,"/cloud_data01/zhengmeisong/wkspace/ol6/gesture/frcnn-gesture/caffe-fast-rcnn/python")
import torch
import torch.nn as nn
import math
from pytorch2caffe import *
import torch.optim as optim
import remove_caffe_model_bn as remove_bn
InWidth = 180
InHeigh = 180
class SmallVgg19(nn.Module):
    def __init__(self, name, in_channels):
        super(SmallVgg19, self).__init__()
        self.op_name = name
        op_list = []
        op_list += [conv_bn_relu(name + "/conv1_1", in_channels, 12, kernel_size=3, stride=2, padding=1),
                    conv_bn_relu(name + "/conv1_2", 12, 12, kernel_size=3, stride=1, padding=1),
                    conv_bn_relu(name + "/conv2_1", 12, 20, kernel_size=3, stride=2, padding=1),
                    conv_bn_relu(name + "/conv2_2", 20, 20, kernel_size=3, stride=1, padding=1),
                    conv_bn_relu(name + "/conv3_1", 20, 40, kernel_size=3, stride=1, padding=1),
                    conv_bn_relu(name + "/conv3_2", 40, 40, kernel_size=3, stride=1, padding=1),
                    conv_bn_relu(name + "/conv3_3", 40, 40, kernel_size=3, stride=1, padding=1),
                    conv_bn_relu(name + "/conv3_4", 40, 40, kernel_size=3, stride=1, padding=1),
                    conv_bn_relu(name + "/conv4_1", 40, 64, kernel_size=3, stride=1, padding=1),
                    conv_bn_relu(name + "/conv4_2", 64, 64, kernel_size=3, stride=1, padding=1),
                    conv_bn_relu(name + "/conv4_3", 64, 64, kernel_size=3, stride=1, padding=1),
                    conv_bn_relu(name + "/conv4_4", 64, 64, kernel_size=3, stride=1, padding=1),
                    conv_bn_relu(name + "/conv5_1", 64, 64, kernel_size=3, stride=1, padding=1),
                    conv_bn_relu(name + "/conv5_2", 64, 64, kernel_size=3, stride=1, padding=1),
                    conv_bn_relu(name + "/conv5_3", 64, 64, kernel_size=3, stride=1, padding=1),
                    conv_bn_relu(name + "/conv5_4", 64, 64, kernel_size=3, stride=1, padding=1),
                    conv(name + "/saliency_map", 64, 1, kernel_size=1, stride=1, padding=0)
                    ]


        op_list += [ op_name(name + '/sigmoid', nn.Sigmoid()) ]
        self.conv_block = nn.Sequential(*op_list)
    
    def forward(self, x):
        feature = self.conv_block(x)

        return feature

    def generate_caffe_prototxt(self, caffe_net, x):
        return generate_caffe_prototxt(self.conv_block, caffe_net, x)

def get_symbol(symbol_name = 'SmallVgg19', input_nc = 3):
    return SmallVgg19(symbol_name, input_nc)

def load_model(model, model_path):
    print ('load SmallVgg19')
    sys.stdout.flush()
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except:
        pretrained_dict = torch.load(model_path, map_location='cpu')
        model_dic = model.state_dict()

        for k,v in pretrained_dict.items():
            if (k in model_dic) and  v.size() == model_dic[k].size():
                model_dic[k] = v
            else:
                print ('Unknow keys:', k)
                sys.exit()
                # key_split = k.split('.')
                # key_split[0] = 'pt_block'
                # key_split[1] = str( int(key_split[1]) - 2 ) 
                # key_new = '.'.join(key_split)
                # if (key_new in model_dic) and (v.size() == model_dic[key_new].size()):
                #     model_dic[key_new] = v
                # else:
                #     print 'Unknow keys:', key_new
                #     sys.exit()
        model.load_state_dict(model_dic)

def pytorch2caffe(proto_path, caffemodel_path):
    model = SmallVgg19('SmallVgg19', 3)
    load_model(model, '../../checkpoints/SmallVgg19_latest.pth')
    model.eval()
    caffe_net = caffe.NetSpec()
    layer = L.Input(shape=dict(dim=[1, 3, InHeigh, InWidth]))
    caffe_net.tops['data'] = layer
    model.generate_caffe_prototxt(caffe_net, layer)
    # print(caffe_net.to_proto())
    with open(proto_path, 'w') as f:
        f.write(str(caffe_net.to_proto()))

    caffe_net = caffe.Net(proto_path, caffe.TEST)
    convert_weight_from_pytorch_to_caffe(model, caffe_net)
    caffe_net.save(caffemodel_path)

    caffe_prototxt_nobn = proto_path.replace('.prototxt', '_nobn.prototxt')
    caffe_caffemodel_nobn = caffemodel_path.replace('.caffemodel', '_nobn.caffemodel')
    remove_bn.zrnRemoveProtoCaffemodelBN(proto_path, caffemodel_path, caffe_prototxt_nobn, caffe_caffemodel_nobn)
    print ('----------converted finished--------------')

    print ('Begin to Test the Model Created!')    
    pytorch_input = torch.randn(1, 3, InHeigh, InWidth)
    pytorch_output = model(pytorch_input)

    caffe.set_mode_cpu()
    caffe_net = caffe.Net(caffe_prototxt_nobn, caffe_caffemodel_nobn, caffe.TEST)
    forward_kwargs = {'data': pytorch_input.numpy()}
    caffe_output = caffe_net.forward(**forward_kwargs).items()[0][1]

    criterion = nn.MSELoss()
    loss = criterion(pytorch_output, torch.FloatTensor(caffe_output))

    print ('RMSE LOSS:', math.sqrt(loss.item()))
    print ('pytorch output:   ', pytorch_output.detach().numpy()[0,:6])
    print ('caffe_nobn output:', caffe_output[0,:6])
    if (math.sqrt(loss.item()) < 0.001):
        print ('Convert Successful!' )
    else:
        print ('Convert failed!' )

    sys.exit()

if __name__=='__main__':
    pytorch2caffe('../../caffemodel/SmallVgg19.prototxt',  '../../caffemodel/SmallVgg19.caffemodel')

    net = SmallVgg19('SmallVgg19', 3)
    net.eval()

    criterion = nn.MSELoss()
    target = torch.randn(1, 6)
    input = torch.randn(1, 3, InHeigh, InWidth)

    output, features = net(input)
    print (output.size())
    sys.exit()

    # optimizer = optim.SGD(net.parameters(), lr=0.01)
    
    # for i in range(1000):
    #     output = net(input)
    #     loss = criterion(output, target)
    #     print loss.item()
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

