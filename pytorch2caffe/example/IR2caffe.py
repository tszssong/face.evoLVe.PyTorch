import sys
sys.path.insert(0, '../')
sys.path.insert(0, '.')
sys.path.insert(0, '../../backbone/')
import torch
from torch.autograd import Variable
from torchvision.models import resnet
import pytorch_to_caffe
from model_irse import IR_18, IR_50
pth='../../../py-model/IR50_ms1m_epoch120.pth'

if __name__ == '__main__':
    # name = 'Res18'
    # net= IR_18()
    # checkpoint = torch.load("/cloud_data01/zhengmeisong/wkspace/olx/py-model/Res18/Res18.pth")
   
    name = pth.replace('.pth','')
    net= IR_50()
    checkpoint = torch.load(pth)

    net.load_state_dict(checkpoint)
    net.eval()
    input = torch.ones([1, 3, 112, 112])
    # input=torch.ones([1,3,224,224])
    pytorch_to_caffe.trans_net(net, input, name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))