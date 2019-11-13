import sys
sys.path.insert(0,'./..')
import torch
from torch.autograd import Variable
from torchvision.models import resnet
import pytorch_to_caffe
import pose_hrnet
import yaml
import pickle

if __name__=='__main__':
    name='pose_hrnet_w32'
    #resnet18=resnet.resnet18()
    f = open('./../w32_256x192_adam_lr1e-3.yaml')
    content = yaml.load(f)

    model = pose_hrnet.get_pose_net(cfg=content, is_train=False)
    checkpoint = torch.load("./../pose_hrnet_w32_256x192.pth")
    model.load_state_dict(checkpoint)
    model.eval()
    #resnet18.load_state_dict(checkpoint)
    #resnet18.eval()
    input=torch.ones([1,3,256,192])
    output=model(input)
    print(output)
     #input=torch.ones([1,3,224,224])

    pytorch_to_caffe.trans_net(model,input,name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))