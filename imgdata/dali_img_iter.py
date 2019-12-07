import os, sys, time, argparse
import random
import os.path
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data

import cv2
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as dali_ops
import nvidia.dali.types as dali_types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

class reader_pipeline(Pipeline):
    def __init__(self, image_dir, batch_size, num_threads, device_id):
        super(reader_pipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = dali_ops.FileReader(file_root = image_dir, random_shuffle = False)
        self.decode = dali_ops.ImageDecoder(device = 'mixed', output_type = dali_types.RGB)
        self.augmentations = {}
        self.augmentations["contrast"] = dali_ops.Contrast(device = "gpu", contrast = 1.5)
        self.augmentations["brightness"] = dali_ops.Brightness(device = "gpu", brightness = 0.5)
        self.augmentations["saturation"] = dali_ops.Saturation(device = "gpu", saturation = 0.2)
        self.augmentations["jitter"] = dali_ops.Jitter(device = "gpu")
        self.augmentations["hue"] = dali_ops.Hue(device = "gpu", hue = 0.)
        self.cmn_img = dali_ops.CropMirrorNormalize(device = "gpu",
                                           crop=(112, 112),  crop_pos_x=0, crop_pos_y=0,
                                           output_dtype = dali_types.FLOAT, image_type=dali_types.RGB,
                                           mean=[0.5*255, 0.5*255, 0.5*255],
                                           std=[0.5*255, 0.5*255, 0.5*255]
                                           )
        self.brightness_change = dali_ops.Uniform(range=(0.5,1.2))
        self.rotate = dali_ops.Rotate(device = "gpu")
        self.rng = dali_ops.Uniform(range = (-50.0,40.0))
        self.rd_bright = dali_ops.Brightness(device="gpu")

        # self.disturb = dali_ops.CoinFlip(probability=0.5)
        self.disturb = random.randint(0,1)
    # def transform(self, inputs):
    #     if 

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        # bright = self.brightness_change()
        # images = self.rd_bright(images, brightness=bright)
        angle = self.rng()
        # images = self.rotate(images, angle = angle)
        _disturb = random.randint(0,1)
        print(type(_disturb))
        print(_disturb)
        if _disturb:
            images = self.rotate(images, angle=angle)
        # images = self.augmentations["brightness"](images, brightness=bright)
        # images = dali_ops.Brightness(device = "gpu", brightness=self.brightness_change())(images)
        # _rd = random.randint(0,1)
        # if _rd == 1:
        #     _cbs = random.randint(0,6)
        #     if _cbs == 0:
        #         images = self.augmentations["contrast"](images) 
        #     elif _cbs == 1:
        #         images = self.augmentations["brightness"](images)
        #     elif _cbs == 2:
        #         images = self.augmentations["saturation"](images)
        #     elif _cbs == 3:
        #         images = self.augmentations["contrast"](images) 
        #         images = self.augmentations["brightness"](images)
        #     elif _cbs ==4:
        #         images = self.augmentations["brightness"](images)
        #         images = self.augmentations["saturation"](images)
        #     elif _cbs == 5:
        #         images = self.augmentations["contrast"](images) 
        #         images = self.augmentations["saturation"](images)
        #     elif _cbs == 6:
        #         images = self.augmentations["contrast"](images)
        #         images = self.augmentations["brightness"](images) 
        #         images = self.augmentations["saturation"](images)
        # _rd = random.randint(0,2)
        # if _rd == 1:
        #     images = self.augmentations["jitter"](images)
        # _rd = random.randint(0,1)
        # if _rd == 1:
        # # if np.random.randint(0,2):
        # # images = self.augmentations["hue"](images)
        #     hue_param = random.randint(0,100)
        #     print(_rd, hue_param)
        #     images = dali_ops.Hue(device = "gpu", hue = 45.)(images)
        imgs = self.cmn_img(images)
        return (imgs, labels)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='/home/ubuntu/zms/data/ms1m_emore_img/')
    parser.add_argument('--input-size', type=str, default="112, 112")
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epoch', type=int, default=25)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--num-classes', type=int, default=143474)
    args = parser.parse_args()
    
    INPUT_SIZE   =  [ int(args.input_size.split(',')[0]), int(args.input_size.split(',')[1]) ]
    GPU_ID = [int(i) for i in args.gpu_ids.split(",") ]
    show_x = 6
    show_y = 3

    train_dir = os.path.join(args.data_root, 'data_100')
    train_pipes = reader_pipeline(train_dir, args.batch_size, args.num_workers, device_id = GPU_ID[0])
    train_pipes.build()
    train_loader = DALIGenericIterator(train_pipes, ['imgs', 'labels'],\
                                       train_pipes.epoch_size("Reader"), \
                                       auto_reset=True)
    a = dali_ops.Uniform(range=(0.5,1.2))
    
    start = time.time()
    show_sample_img = np.zeros((show_y*INPUT_SIZE[1], show_x*INPUT_SIZE[0], 3), dtype=np.uint8)
    x=0
    y=0
    for epoch in range(10):
        print("epoch %d"%epoch)
        # for inputs, labels in iter(train_loader):
        for i, datas  in enumerate(train_loader):
           
            inputs = datas[0]['imgs']
            labels = datas[0]['labels']
            labels = labels.view(args.batch_size)

            inputs = inputs.cpu().numpy()
            labels = labels.cpu().numpy()
            for b_idx in range(inputs.shape[0]):
                im = inputs[b_idx]
                label = labels[b_idx]
                im = im*127.5 + 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1,2,0))
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                # print(b_idx, ":", im.shape, label)
                # cv2.imshow("im decode:", im)
                # cv2.waitKey(100)
                show_sample_img[y*INPUT_SIZE[1]:(y+1)*INPUT_SIZE[1], 
                                x*INPUT_SIZE[0]:(x+1)*INPUT_SIZE[0],:] = im
                x = x+1
                if x==show_x:
                    y = y+1
                    x = 0
                    if y==show_y:
                        y = 0
                        cv2.imshow("sample", show_sample_img)
                        cv2.waitKey(1000)
                        # cv2.imwrite("./sample%d.jpg"%epoch, show_sample_img)

    print("10 epoch use time: %.2f s"%(time.time()-start))