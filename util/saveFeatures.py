import os, sys, time, os.path, argparse, socket
import numpy as np
import bisect
import PIL
from PIL import Image 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import torch.nn.functional as F
import torchvision
from torchvision import transforms
sys.path.append( os.path.join( os.path.dirname(__file__),'../backbone/') )
from model_resnet import ResNet_50, ResNet_101, ResNet_152
from model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152


def alignedImg2feature(img_root, save_root, model, device="cpu", suffix='.ft', start=1015000, end=1015099):
    sp_savepath = save_root + '/spots/' 
    id_savepath = save_root + '/ids/'
    if not os.path.isdir(sp_savepath):
        os.makedirs(sp_savepath)
    if not os.path.isdir(id_savepath):
        os.makedirs(id_savepath)
    print(img_root)
    done_count=0
    with open(save_root + '/ft_lst', 'w') as fw:
        for idx in range(start, end):
            img_name = str(idx) + '.jpg'
            ft_name  = img_name.replace('.jpg', suffix)
            try:
                sp_img = Image.open(img_root + '/spots/' + img_name).convert('RGB')
                id_img = Image.open(img_root + '/ids/' + img_name).convert('RGB')
            except:
                print(img_name, "not exists")
                continue
            sp_img = test_transform(sp_img)
            id_img = test_transform(id_img)
            assert (sp_img.size()[1] == 112) and (sp_img.size()[2] == 112)
            batchIn = torch.empty(2,3,112,112)
            batchIn[0] = sp_img
            batchIn[1] = id_img
            with torch.no_grad():
                batchFea = model(batchIn.to(device))
            batchFea = F.normalize(batchFea).detach()
            sp_feat = batchFea[0].cpu().numpy()
            id_feat = batchFea[1].cpu().numpy()
            np.savetxt(sp_savepath + ft_name, sp_feat)
            np.savetxt(id_savepath + ft_name, id_feat)
            done_count += 1
            fw.write(ft_name+'\n')
    return done_count

def alignedImg2feature_vlist(img_list, save_root, model, device="cpu", suffix='.ft'):
    
    print(img_list)
    done_count=0
    with open(img_list, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            path = line.strip().split(' ')[0]
            img_name  = path.split('/')[-1]
            sub_dir   = path.split('/')[-2]
            ft_subdir = save_root + '/' + sub_dir
            if not os.path.isdir(ft_subdir):
                os.makedirs(ft_subdir)
            ft_name  = img_name.replace('.jpg', suffix)
            try:
                img = Image.open(path).convert('RGB')
            except:
                print(path, "not exists")
                continue
            img = test_transform(img)
            assert (img.size()[1] == 112) and (img.size()[2] == 112)
            batchIn = torch.empty(1,3,112,112)
            batchIn[0] = img
            
            with torch.no_grad():
                batchFea = model(batchIn.to(device))
            batchFea = F.normalize(batchFea).detach()
            feature = batchFea[0].cpu().numpy()
            
            np.savetxt(ft_subdir + '/' + ft_name, feature)
            if(done_count%500==0):
                print(done_count,"ft extracted")
            done_count += 1
    return done_count
# extra features, to be compatible with mxnet
# test: hpcgpu40:/cloud_data01/zhengmeisong/wkspace/qh_recog_clean/deploy_st/getP5Casia/run_test_torch.sh
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='/cloud_data01/StrongRootData/TestData/CASIA-IvS-Test/')
    parser.add_argument('--list', type=str, default='CASIA-IvS-Test-final-v3-revised.lst')
    parser.add_argument('--ft-suffix', type=str, default='.ft')
    parser.add_argument('--backbone-resume-root', type=str, default='../../models/py/r50e1b12000_082509/')
    parser.add_argument('--backbone-name', type=str, default='ResNet_50') # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    parser.add_argument('--input-size', type=str, default="112, 112")
    parser.add_argument('--embedding-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=120)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--gpu-ids', type=str, default='1')
    
    args = parser.parse_args()
    
    INPUT_SIZE = [ int(args.input_size.split(',')[0]), int(args.input_size.split(',')[1]) ]
    BACKBONE = eval(args.backbone_name)(INPUT_SIZE)
    backbone_load_path = args.backbone_resume_root + args.backbone_resume_root.split('/')[-2] + '.pth'
    if backbone_load_path and os.path.isfile(backbone_load_path):
        print("Loading Backbone Checkpoint '{}'".format(backbone_load_path))
        BACKBONE.load_state_dict(torch.load(backbone_load_path)) 
    else:
        print("No Checkpoint Error!" )
    test_transform = transforms.Compose([ transforms.ToTensor(), 
                     transforms.Normalize(mean =  [0.5, 0.5, 0.5], std =  [0.5, 0.5, 0.5]), ])

    DEVICE =  torch.device("cuda:%d"%(int(args.gpu_ids)) if torch.cuda.is_available() else "cpu")
    BACKBONE = BACKBONE.to(DEVICE)
    BACKBONE.eval()
    
    start = time.time()
    ftSaveRoot = args.backbone_resume_root + args.data_root.split('/')[-2]
    print("save ft to ", ftSaveRoot)
    if not os.path.isdir(ftSaveRoot):
        os.makedirs(ftSaveRoot)
    num_processed = alignedImg2feature_vlist( args.data_root+args.list, ftSaveRoot, \
                                              BACKBONE, DEVICE, suffix=args.ft_suffix)
    print(num_processed, "pairs of feature saved!")
    print("extra feature used %.2f s"%(time.time()-start) )

