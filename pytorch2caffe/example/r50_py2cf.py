import sys

sys.path.insert(0, '../')
sys.path.insert(0, '.')
import torch
from torch.autograd import Variable
from torchvision.models import resnet
import pytorch_to_caffe
from model_resnet import ResNet_50

if __name__ == '__main__':
    name = 'ResNet_50'
    net= ResNet_50()
    checkpoint = torch.load("/cloud_data01/zhengmeisong/wkspace/olx/py-model/ResNet_50_Epoch_35.pth")

    net.load_state_dict(checkpoint)
    net.eval()
    input = torch.ones([1, 3, 112, 112])
    # input=torch.ones([1,3,224,224])
    pytorch_to_caffe.trans_net(net, input, name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))