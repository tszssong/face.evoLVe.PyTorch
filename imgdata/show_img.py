import os, sys, time, cv2
from PIL import Image
import numpy as np

def showBatch(inputs, labels, features=None, show_x=12, show_y=3, use_hard = True):
    im_width = inputs.shape[3]   #0,1,2,3
    im_heigh = inputs.shape[2]   #n,c,w,h
    tri_batch_size = inputs.shape[0]/3
    show_x = min(int(inputs.shape[0]/3), show_x)
    if not features is None:
        # print("showBatch:", inputs.shape, features.shape)
        assert inputs.shape[0] == features.shape[0]
        if(use_hard):
            ind_list = [idx*3 for idx in range(int(show_x))]
            indices = np.array(ind_list)
            anchor = np.take(features, indices, axis=0)
            indices = indices + 1
            positive = np.take(features, indices, axis=0)
            indices = indices + 1
            negative = np.take(features, indices, axis=0)
        else:
            anchor   = features[0:show_x,:]
            positive = features[show_x:2*show_x,:]
            negative = features[2*show_x:3*show_x,:]
        dp = np.sum( np.power((anchor - positive), 2), axis=1 )
        dn = np.sum( np.power((anchor - negative), 2), axis=1)
        # print("showBatch:", dp, dn)
    show_sample_img = np.zeros( (show_y*im_width, show_x*im_heigh, 3), dtype=np.uint8)
    x=0
    y=0
    for x in range(show_x):
        for y in range(show_y):
            if use_hard:
                idx = x*show_y + y
            else:
                idx = int(y*tri_batch_size + x)
            im = inputs[idx]
            label = labels[idx]
            im = im*127.5 + 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1,2,0))
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            im = cv2.putText(im, str(label), (2,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            if( (not features is None) and (y==1) ):
                im = cv2.putText(im, str(dp[x]), (2,80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255,0), 2)
            if( (not features is None) and (y==2)):
                im = cv2.putText(im, str(dn[x]), (2,80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255,0), 2)
            show_sample_img[y*im_width:(y+1)*im_width, 
                            x*im_heigh:(x+1)*im_heigh,:] = im
    cv2.imshow("sample", show_sample_img)
    cv2.waitKey()   
                     
def saveBatch(inputs, labels, features=None, show_x=12, show_y=3, batchIdx = 1):
    if not os.path.isdir('../tmp'):
        os.makedirs('../tmp')
    im_width = inputs.shape[3]   #0,1,2,3
    im_heigh = inputs.shape[2]   #n,c,w,h
    tri_batch_size = inputs.shape[0]/3
    show_x = min(int(inputs.shape[0]/3), show_x)
    if not features is None:
        # print("showBatch:", inputs.shape, features.shape)
        assert inputs.shape[0] == features.shape[0]
        ind_list = [idx*3 for idx in range(int(show_x))]
        indices = np.array(ind_list)
        anchor = np.take(features, indices, axis=0)
        indices = indices + 1
        positive = np.take(features, indices, axis=0)
        indices = indices + 1
        negative = np.take(features, indices, axis=0)
       
        dp = np.sum( np.power((anchor - positive), 2), axis=1 )
        dn = np.sum( np.power((anchor - negative), 2), axis=1)
        # print("showBatch:", dp, dn)
    show_sample_img = np.zeros( (show_y*im_width, show_x*im_heigh, 3), dtype=np.uint8)
    x=0
    y=0
    for x in range(show_x):
        for y in range(show_y):
            idx = x*show_y + y
           
            im = inputs[idx]
            label = labels[idx]
            im = im*127.5 + 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1,2,0))
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            im = cv2.putText(im, str(label), (2,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            if( (not features is None) and (y==1) ):
                im = cv2.putText(im, str(dp[x]), (2,80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255,0), 2)
            if( (not features is None) and (y==2)):
                im = cv2.putText(im, str(dn[x]), (2,80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255,0), 2)
            show_sample_img[y*im_width:(y+1)*im_width, 
                            x*im_heigh:(x+1)*im_heigh,:] = im
    cv2.imwrite('../tmp/%d.jpg'%(batchIdx), show_sample_img)  