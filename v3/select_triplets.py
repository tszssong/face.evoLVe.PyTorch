import torch
import numpy as np
import random
import time
#2-1;3-3;4-6;5-10;6-15;7-21;8-28;9-36;10-45;11-55
d_lut = {2:50,3:51,4:8,5:5,6:4,7:3,8:2,9:2}
def get_num_per_id(labels):
    label_dict = {}
    for id in labels:
        if id in label_dict:
            label_dict[id] += 1
        else:
            label_dict[id] = 1
    return label_dict

def select_triplets(features, baglabel_1v, bagSize, id_per_batch=40,margin = 0.5):
    assert features.shape[0] == bagSize
    label_dict = get_num_per_id(baglabel_1v)
    nCount = 0
    bagList = []
    last_label = 30000000 
    for a_idx in range( bagSize ):
        
        a_label = baglabel_1v[a_idx]
        if a_label != last_label:
            last_label = a_label
            id_count = 0
            num_id  = label_dict[a_label]
            if num_id < 2:
                continue
            elif num_id < 10:
                num_rep = d_lut[num_id]
            else:
                num_rep = 1
        else:
            id_count += 1 

        if id_count == num_id:
            continue

        dist = 1 - 2*torch.mm(features[a_idx].reshape(1,-1), features.t())
        n_dists = dist.numpy()[0]
        
        for idx in range(num_id-id_count):
            p_idx = a_idx + idx
            p_dist = dist.numpy()[0][p_idx]
            n_dists[ np.where(baglabel_1v==a_label) ] = -8192 #np.NaN    #fill same ids with a bigNumber
            m = margin*margin
            n_dists = n_dists - p_dist - m
            n_candidates = np.where(np.logical_and(n_dists>0, n_dists>p_dist))[0]
            if n_candidates.shape[0] < 1:
                continue
            elif n_candidates.shape[0] > num_rep:
                n_idxs = np.random.choice(n_candidates, num_rep)
            else:
                n_idxs = n_candidates
            for n_idx in n_idxs:
                bagList.append((a_idx,p_idx,n_idx))
                nCount += 1
    return bagList, nCount
