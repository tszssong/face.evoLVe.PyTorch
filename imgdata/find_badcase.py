import os, sys, time
import os.path
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
import cv2
import torch.nn.functional as F
sys.path.append( os.path.join( os.path.dirname(__file__),'../backbone/') )
from model_resnet import ResNet_50, ResNet_101, ResNet_152

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
def jpeg4py_loader(path):
    with open(path, 'rb') as f:
        img = jpeg.JPEG(f).decode()
        return Image.fromarray(img)  #TODO: torchvison transpose use PIL img

def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images

class TripletHardImgData(data.Dataset):
    def __init__(self, root, model, transform=None, target_transform=None):
        super(TripletHardImgData, self).__init__()
        self.root = root
        self.transform = transform               # transform datas
        self.target_transform = target_transform # transform targets
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions=IMG_EXTENSIONS)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples                #samples is a list like below:
        #[(path1,id1),(path2,id1),(path3,id1), (path4,id2),(path5,id2).....]
        self.targets = [s[1] for s in samples] # targets is a list with ids
        print("samples:", len(self.samples))

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):   # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        pos_indexes = np.where( np.array(self.targets)==target )[0]
        pos_indexes = np.delete( pos_indexes, np.where(pos_indexes==index))
        if pos_indexes.shape[0] == 0:
            positive_idx = index        # in case some id only 1 image
        else:
            positive_idx = np.random.choice( pos_indexes )
        # print(index, positive_idx)
        
        positive_path,pos_label = self.samples[positive_idx]   
        negative_idx = np.random.choice( np.where( np.array(self.targets)!=target)[0] )
        negative_path,neg_label = self.samples[negative_idx]   
        # print("label:", target, pos_label, neg_label)
        # print("idx:", target, positive_idx, negative_idx)
        # print("pth:", path, positive_path, negative_path)
        img1 = pil_loader(path)
        img2 = pil_loader(positive_path)
        img3 = pil_loader(negative_path)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), (target, pos_label, neg_label)

    def __len__(self):
        return len(self.samples)


if __name__=='__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    RESIZE_SCALE = [1.0, 1.0]
    INPUT_SIZE   = [112, 112]       # support: [112, 112] and [224, 224]
    RGB_MEAN     = [0.5, 0.5, 0.5]  # for normalize inputs to [-1, 1]
    RGB_STD      = [0.5, 0.5, 0.5]
    DATA_ROOT = '/home/ubuntu/zms/data/dl2dl3/'
    DATA_ROOT = '/media/ubuntu/9a42e1da-25d8-4345-a954-4abeadf1bd02/home/ubuntu/song/data/dl2dl3w1/'
    show_x = 32
    show_y = 3
    model = ResNet_50(INPUT_SIZE)
    embedding_size = 512
    BACKBONE_RESUME_ROOT = '/home/ubuntu/zms/models/ResNet_50_Epoch_33.pth'
    bd_save_pos = '/home/ubuntu/zms/data/badcases/pos/'
    bd_save_neg = '/home/ubuntu/zms/data/badcases/neg/'
    if not os.path.isdir(bd_save_pos):
        os.makedirs(bd_save_pos)
    if not os.path.isdir(bd_save_neg):
        os.makedirs(bd_save_neg)
    # plists = open('../../py-log/pos.lst', 'w')
    # nlists = open('../../py-log/neg.lst', 'w')
    if os.path.isfile(BACKBONE_RESUME_ROOT):
        print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
        model.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
    else:
        print("model file does not exists!!!")

    # train_transform = transforms.Compose([ 
    #     transforms.Resize([int(RESIZE_SCALE[0]*INPUT_SIZE[0]), int(RESIZE_SCALE[1]*INPUT_SIZE[0])]), # smaller side resized
    #     transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean = RGB_MEAN, std = RGB_STD),
    # ])
    train_transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN, std = RGB_STD),
    ])
    dataset_train = TripletHardImgData(os.path.join(DATA_ROOT, 'imgs'), model, train_transform)
    # dataset_train = TripletHardImgData(os.path.join(DATA_ROOT, 'imgs'), model, None)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size = show_x, shuffle=True, sampler = None,
        pin_memory = True, num_workers = 4, drop_last = True
    )
    start = time.time()
    show_sample_img = np.zeros((show_y*INPUT_SIZE[1], show_x*INPUT_SIZE[0], 3), dtype=np.uint8)
    x=0
    y=0
    model.eval()
    embeddings = np.zeros([show_x*3, embedding_size])
    p_count = 0
    n_count = 0
    for epoch in range(1):
        print("epoch %d"%epoch)
        for inputs, labels in iter(train_loader):
            a = inputs[0]
            p = inputs[1]
            n = inputs[2]
            inputs = torch.cat((a,p,n), 0) 

            network_in = inputs.to(DEVICE).cpu()
            features = model(network_in)
            features = F.normalize(features)
            anchor   = features[0:show_x,:]
            positive = features[show_x:2*show_x,:]
            negative = features[2*show_x:3*show_x,:]
            d_pos = (anchor - positive).pow(2).sum(1)
            d_neg = (anchor - negative).pow(2).sum(1)
            # print(features.size())
            dp = d_pos.detach().numpy()
            dn = d_neg.detach().numpy()
            # print("p_dist:", np.average(dp))
            # print("n_dist:", np.average(dn))
            TH = (np.average(dp) + np.average(dn))/2.0
            alpha = 0.3
            bd_p_idx = np.where( dp > TH-0.3 )[0]
            bd_n_idx = np.where( dn < TH+0.3 )[0]
            print(TH)
            if  (bd_n_idx.shape[0]==0)  and  (bd_p_idx.shape[0]==0):
                continue
            inputs = inputs.numpy()
            a_label = labels[0]
            p_label = labels[1]
            n_label = labels[2]
            labels = torch.cat((a_label, p_label, n_label), 0)
            network_label = labels.to(DEVICE).long()
            labels = labels.numpy()

            for p_idx in  range( bd_p_idx.shape[0]  ):
                print("bd pos:", TH, bd_p_idx )
                p_count = p_count + 1
                im = inputs[ bd_p_idx[p_idx] ]
                im = im*127.5 + 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1,2,0))
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                imp = inputs[ bd_p_idx[p_idx]+show_x]
                imp = imp*127.5 + 127.5
                imp = imp.astype(np.uint8)
                imp = np.transpose(imp, (1,2,0))
                imp = cv2.cvtColor(imp, cv2.COLOR_RGB2BGR)
                # savedir = bd_save_pos + str(p_count) + '/'
                # if not os.path.isdir(savedir):
                #     os.makedirs(savedir)
                cv2.imwrite(bd_save_pos + str(n_count) +'_a.jpg', im)
                cv2.imwrite(bd_save_pos + str(n_count) +'_p.jpg', imp)
            for n_idx in  range( bd_n_idx.shape[0]  ):
                print("bd neg:", TH, bd_n_idx )
                n_count = n_count + 1
                im = inputs[ bd_n_idx[n_idx] ]
                im = im*127.5 + 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1,2,0))
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                imp = inputs[ bd_n_idx[n_idx]+show_x*2 ]
                imp = imp*127.5 + 127.5
                imp = imp.astype(np.uint8)
                imp = np.transpose(imp, (1,2,0))
                imp = cv2.cvtColor(imp, cv2.COLOR_RGB2BGR)
                # savedir = bd_save_neg + str(n_count) + '/'
                # if not os.path.isdir(savedir):
                #     os.makedirs(savedir)
                cv2.imwrite(bd_save_neg + str(n_count) +'_a.jpg', im)
                cv2.imwrite(bd_save_neg + str(n_count) +'_n.jpg', imp)
            # for b_idx in range(inputs.shape[0]):
            #     im = inputs[b_idx]
            #     label = labels[b_idx]
            #     im = im*127.5 + 127.5
            #     im = im.astype(np.uint8)
            #     im = np.transpose(im, (1,2,0))
            #     im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            #     im = cv2.putText(im, str(label), (2,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255), 2)
            #     if(y==1):
            #         im = cv2.putText(im, str(dp[x]), (2,80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255,0), 2)
            #     if(y==2):
            #         im = cv2.putText(im, str(dn[x]), (2,80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255,0), 2)
                        
            #     show_sample_img[y*INPUT_SIZE[1]:(y+1)*INPUT_SIZE[1], 
            #                     x*INPUT_SIZE[0]:(x+1)*INPUT_SIZE[0],:] = im
            #     x = x+1
            #     if x==show_x:
            #         y = y+1
            #         x = 0
            #         if y==show_y:
            #             y = 0
            #             cv2.imshow("sample", show_sample_img)
            #             cv2.waitKey()

    print("1 epoch use time: %.2f s"%(time.time()-start))
