import os, sys
import argparse  
import cv2
import numpy as np

from util.utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy

def extractPics(carry, carry_issame, save_path):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for idx in range(carry.shape[0]):
        print(idx)
        im = carry[idx]
        print(im.shape, im.dtype)
        im = im*127.5 + 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1,2,0))
        print(im.shape, im.dtype)
        cv2.imwrite(save_path + '/' + str(idx) + '.jpg', im)
        
    print(carry.shape, len(carry_issame))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='get verification images')
    parser.add_argument('--veri-root', default='/data03/zhengmeisong/data/ms1m_emore_img/', help='blps read form')
    parser.add_argument('--save-root', default='/data03/zhengmeisong/data/verifyDatas/', help='images write to')
    args = parser.parse_args()

    lfw, cfp_ff, cfp_fp, agedb, calfw, cplfw, vgg2_fp, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_issame, calfw_issame, cplfw_issame, vgg2_fp_issame = get_val_data(args.veri_root)

    extractPics(cfp_fp, cfp_fp_issame, args.save_root + '/cfp_fp/')

