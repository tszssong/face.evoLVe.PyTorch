import torch
import numpy as np
import random
import time,sys
#2-1;3-3;4-6;5-10;6-15;7-21;8-28;9-36;10-45;11-55
d_lut = {2:6,3:3,4:2,5:1,6:1,7:1}
def get_num_per_id(labels):
    label_dict = {}
    for id in labels:
        if id in label_dict:
            label_dict[id] += 1
        else:
            label_dict[id] = 1
    return label_dict

def select_triplets(features, baglabel_1v, bagSize, id_per_batch=10,margin = 0.5,device="cpu"):
    mmTimes = 0.0
    copyTimes = 0.0
    t1 = time.time()
    features = features.to(device)
    np.set_printoptions(suppress=True)
    assert features.shape[0] == bagSize
    label_dict = get_num_per_id(baglabel_1v)
    baglabel_1v = torch.tensor(baglabel_1v)
    nCount = 0
    bagList = []
    last_label = 30000000 
    for a_idx in range( bagSize ):
        p_skip = 1
        a_label = baglabel_1v[a_idx]
        a_label = int(a_label)
        num_id  = label_dict[a_label]

        if a_label != last_label:
            last_label = a_label
            aid_count = 1
        else:
            aid_count += 1 
        if aid_count == num_id:
            continue

        num_rep = 1
        if num_id < 2:
            continue
        elif num_id <= 7:
            num_rep = d_lut[num_id]
        elif num_id > 7:
            p_skip = round( (num_id*(num_id-1)/2.0) / id_per_batch )

        # print("%d,%d,%d,%d"%(a_idx, num_id, aid_count,p_skip))
        a_fea = features[a_idx].reshape(1,-1)
        t_mm = time.time()
        aTf = torch.mm(a_fea, features.t())
        dist = 2 - 2*aTf
        mmTimes += (time.time()-t_mm)
        t_copy = time.time()
        #dist = dist.cpu()
        copyTimes += (time.time() - t_copy)
         
        #n_dists = dist.numpy()[0].copy()
        #n_dists[ np.where(baglabel_1v==a_label)[0] ] = -8192    #np.NaN    #fill same ids with a bigNumber
#        n_dists = torch.where(baglabel_1v==a_label, torch.tensor(-8192.0), dist)    #np.NaN    #fill same ids with a bigNumber
        n_dists = torch.where(baglabel_1v.to(device)==torch.tensor(a_label).to(device), torch.tensor(-8192.0).to(device), dist.to(device))    #np.NaN    #fill same ids with a bigNumber
       
        # for idx in range(1, num_id-aid_count+1, p_skip):
        for idx in range(1, num_id-aid_count+1):
            tmp = random.randint(1,p_skip)
            # if(p_skip>1):
            #     print("%d: %d"%(num_id,tmp),'-',p_skip,'-',aid_count,' '),
            #     sys.stdout.flush()
            if (tmp > 1):
                continue 
            p_idx = a_idx + idx
            # if (np.random.randint(1,num_id))
            #p_dist = dist.numpy()[0][p_idx]
            p_dist = dist[0][p_idx]
            if(p_dist>1.3):
                continue
            thresh = p_dist + margin
            thresh = torch.tensor(thresh).to(device)
            a = n_dists<thresh
            b = n_dists>p_dist
            n_candidates = a*b
            n_candidates = n_candidates.reshape(n_candidates.shape[1])
            #n_candidates = np.where( np.logical_and( n_dists<thresh, n_dists>p_dist ) )[0]
            
            n_candidates = n_candidates.cpu()
            if n_candidates.shape[0] < 1:
                continue
            elif n_candidates.shape[0] > num_rep:
                n_idxs = np.random.choice(n_candidates, num_rep)
            else:
                n_idxs = n_candidates
            for n_idx in n_idxs:
                bagList.append((a_idx,p_idx,n_idx))
                #bagList.append((a_idx,p_idx,n_idx))
                nCount += 1
    
    print("select time:%.5f"%(time.time()-t1))
    print("mm time:%.5f"%mmTimes)
    print("copy time:%.5f"%copyTimes)
    sys.stdout.flush()
    return bagList, nCount
