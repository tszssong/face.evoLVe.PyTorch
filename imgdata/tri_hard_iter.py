import os, sys, time, cv2, random
import os.path
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
import argparse
import torch.nn.functional as F
sys.path.append( os.path.join( os.path.dirname(__file__),'../backbone/') )
sys.path.append( os.path.join( os.path.dirname(__file__),'../imgdata/') )
from model_resnet import ResNet_50, ResNet_101, ResNet_152
from show_img import showBatch
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
    def __init__(self, root, input_size, \
                 transform=None, target_transform=None, use_list = True):
        super(TripletHardImgData, self).__init__()
        print("triplet hard image dataloader inited")
        self.transform = transform               # transform datas
        self.target_transform = target_transform # transform targets
        self.input_size = input_size
        if use_list:
            samples = self._read_paths(root)
        else:
            classes, class_to_idx = self._find_classes(self.root)
            samples = make_dataset(root, class_to_idx, extensions=IMG_EXTENSIONS)
            if len(samples) == 0:
                raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                    "Supported extensions are: " + ",".join(extensions)))
            self.class_to_idx = class_to_idx
            self.classes = classes
        # random.shuffle(samples)
        self.samples = samples                #samples is a list like below:
        self.targets = [s[1] for s in samples] # targets is a list with ids
       
    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):   # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _read_paths(self, path):
        samples = []
        with open(path, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                [path, id] = line.strip().split(' ')
                samples.append( ( path, int(id) ) )    # match torchvison Folder fomat 
        return samples 

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(target)
        
    def __len__(self):
        return len(self.samples)

if __name__=='__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='triplet image iter')
    parser.add_argument('--data-root', type=str, default='/home/ubuntu/zms/data/ms1m_emore100/')
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--image-size', default=[112, 112])
    parser.add_argument('--model-path', type=str, default='/home/ubuntu/zms/models/ResNet_50_Epoch_33.pth')
    parser.add_argument('--bag-size', type=int, default=120)
    args = parser.parse_args()
    im_width = args.image_size[0]
    im_heigh = args.image_size[1]
    print("triplet hard image iter size:", im_width, im_heigh)

    model = ResNet_50([im_width, im_heigh])
    embedding_size = 512
    if os.path.isfile(args.model_path):
        print("Loading Backbone Checkpoint '{}'".format(args.model_path))
        model.load_state_dict(torch.load(args.model_path))
    else:
        print("model file does not exists!!!")

    train_transform = transforms.Compose([ 
        transforms.Resize([112,112]), # smaller side resized
        transforms.RandomCrop([ im_width, im_heigh ]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5] ),
    ])
    
    # dataset_train = TripletHardImgData(os.path.join(args.data_root, 'imgs'), model, transform=train_transform, use_list=False)
    dataset_train = TripletHardImgData( os.path.join(args.data_root, 'imgs.lst'), model, \
                    batch_size = args.batch_size, bag_size = args.bag_size, input_size = [112,112], \
                    transform=train_transform, use_list=True)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size = args.batch_size, shuffle=True, sampler = None,
        pin_memory = True, num_workers = 4, drop_last = True
    )
    start = time.time()
    model.eval()
    model.to(DEVICE)
    embeddings = np.zeros([args.batch_size*3, embedding_size])
    for epoch in range(600000):
        print("epoch %d"%epoch)
        dataset_train.reset(model, DEVICE)
        for inputs, labels in iter(train_loader):
            a = inputs[0]
            p = inputs[1]
            n = inputs[2]
            inputs = torch.cat((a,p,n), 0) 
            a_label = labels[0]
            p_label = labels[1]
            n_label = labels[2]
            labels = torch.cat((a_label, p_label, n_label), 0)
            network_label = labels.to(DEVICE).long()
            network_in = inputs.to(DEVICE)
            features = model(network_in)
            features = F.normalize(features).detach()

            anchor   = features[0:args.batch_size,:]
            positive = features[args.batch_size:2*args.batch_size,:]
            negative = features[2*args.batch_size:3*args.batch_size,:]
            d_pos = (anchor - positive).pow(2).sum(1)
            d_neg = (anchor - negative).pow(2).sum(1)
            dp = d_pos.cpu().numpy()
            dn = d_neg.cpu().numpy()

            # print("average p_dist:", np.average(dp))
            # print("average n_dist:", np.average(dn))
           
            inputs = inputs.cpu().numpy()
            labels = labels.cpu().numpy()
            features = features.cpu().numpy()
            showBatch(inputs, labels, features, args.batch_size, show_y=3)

    print("10 epoch use time: %.2f s"%(time.time()-start))