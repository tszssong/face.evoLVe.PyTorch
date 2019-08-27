import os, sys, time, os.path, argparse, socket
import numpy as np
import bisect
import PIL
from PIL import Image 

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import sklearn
from scipy import interpolate
from scipy.spatial.distance import pdist

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

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds = 10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    print(nrof_pairs)
    print(actual_issame)
  
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits = nrof_folds, shuffle = False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
   
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    # dist = pdist(np.vstack([embeddings1, embeddings2]), 'cosine')
    print(dist)
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
#         print('best_threshold_index', best_threshold_index, acc_train[best_threshold_index])
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                    dist[test_set],
                                                                                    actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds = 10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits = nrof_folds, shuffle = False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind = 'slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

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

def load_features(load_root, maxNum=70000):
    with open(load_root + '/ft_lst','r') as fread:
        lines = fread.readlines()
        print(len(lines), lines[1])
        issame = []
        emb1 = None 
        emb2 = None
        for line in lines:             # line: 1015074.ft 
            emb_sp = np.loadtxt(load_root + '/spots/' + line.strip())
            emb_id = np.loadtxt(load_root + '/ids/' + line.strip())
           
            if emb1 is None:
                emb1 = emb_sp[np.newaxis,:]
                emb2 = emb_id[np.newaxis,:]
                issame.append(True)
            else:
                emb1 = np.concatenate((emb1, emb_sp[np.newaxis,:]))
                emb2 = np.concatenate((emb2, emb_id[np.newaxis,:]))
                issame.append(True)
         
            ft_name, suffix = line.strip().split('.')
            neg_name = str(int(ft_name) + 1000)
            try:
                emb_neg_id = np.loadtxt(load_root + '/ids/' + neg_name + '.' + suffix)
            except:
                print(neg_name, "not found, skip this one")
                continue
            emb1 = np.concatenate((emb1, emb_sp[np.newaxis,:]))
            emb2 = np.concatenate((emb2, emb_neg_id[np.newaxis,:]))
            issame.append(False) 
            if(len(issame)/2>maxNum): 
                return emb1, emb2, issame   
    return emb1, emb2, issame

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='/data02/zhengmeisong/testData/ja-ivs')
    parser.add_argument('--ft-suffix', type=str, default='.ft')
    parser.add_argument('--backbone-resume-root', type=str, default='/data02/zhengmeisong/models/py/r50e1b12000_082509/')
    parser.add_argument('--backbone-name', type=str, default='ResNet_50') # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    parser.add_argument('--input-size', type=str, default="112, 112")
    parser.add_argument('--embedding-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=120)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--gpu-ids', type=str, default='1')
    parser.add_argument('--start', type=int, default=1000000)
    parser.add_argument('--end',   type=int, default=1069999)
    args = parser.parse_args()
    assert args.start>=1000000 and args.end<1070000 and args.start<args.end
    INPUT_SIZE = [ int(args.input_size.split(',')[0]), int(args.input_size.split(',')[1]) ]
    BACKBONE = eval(args.backbone_name)(INPUT_SIZE)
    backbone_load_path = args.backbone_resume_root +'/'+ args.backbone_resume_root.split('/')[-2] + '.pth'
    # if backbone_load_path and os.path.isfile(backbone_load_path):
    #     print("Loading Backbone Checkpoint '{}'".format(backbone_load_path))
    #     BACKBONE.load_state_dict(torch.load(backbone_load_path)) 
    # else:
    #     print("No Checkpoint Error!" )
    # test_transform = transforms.Compose([ transforms.ToTensor(), 
    #                  transforms.Normalize(mean =  [0.5, 0.5, 0.5], std =  [0.5, 0.5, 0.5]), ])

    # DEVICE =  torch.device("cuda:%d"%(int(args.gpu_ids)) if torch.cuda.is_available() else "cpu")
    # BACKBONE = BACKBONE.to(DEVICE)
    # BACKBONE.eval()
    
    # start = time.time()
    # num_processed = alignedImg2feature( args.data_root, args.backbone_resume_root, \
    #                  BACKBONE, DEVICE, suffix=args.ft_suffix, start=args.start, end=args.end )
    # print(num_processed, "pairs of feature saved!")
    # print("extra feature used %.2f s"%(time.time()-start) )

    start = time.time()
    embedding1, embedding2, issame = load_features(args.backbone_resume_root, 50000)
    print("load feature  used %.2f s"%(time.time()-start) )
    start = time.time()
    thresholds = np.arange(0, 4, 0.0001)
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, embedding1, embedding2, np.asarray(issame))
    print("cal roc used %.2f s"%(time.time()-start) )
    print(tpr)
    print(fpr)   
    print("acc:",accuracy.mean(),"@ TH=",best_thresholds) 
    tar_01 = tpr[bisect.bisect_left(fpr, 0.01)]
    tar_001 = tpr[bisect.bisect_left(fpr, 0.001)]
    tar_0001 = tpr[bisect.bisect_left(fpr, 0.0001)]
    tar_00001 = tpr[bisect.bisect_left(fpr, 0.00001)]
    tar_000001 = tpr[bisect.bisect_left(fpr, 0.000001)]
    print(tar_01, tar_001, tar_0001, tar_00001, tar_000001)
    print("%.5f @ FPR=%.7f"%(tar_01,0.01))
    print("%.5f @ FPR=%.7f"%(tar_001,0.001))
    print("%.5f @ FPR=%.7f"%(tar_0001,0.0001))
    print("%.5f @ FPR=%.7f"%(tar_00001,0.00001))
    print("%.5f @ FPR=%.7f"%(tar_000001,0.000001))
