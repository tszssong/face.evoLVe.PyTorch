import os, sys, time, cv2
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
    def __init__(self, root, model, transform=None, target_transform=None, use_list = True):
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
        
        positive_path,pos_label = self.samples[positive_idx]   
        negative_idx = np.random.choice( np.where( np.array(self.targets)!=target)[0] )
        negative_path,neg_label = self.samples[negative_idx]  

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

def showBatch(inputs, labels, features=None, show_x=12, show_y=3):
    im_width = inputs.shape[3]   #0,1,2,3
    im_heigh = inputs.shape[2]   #n,c,w,h
    show_x = min(int(inputs.shape[0]/3), show_x)
    if not features is None:
        print("showBatch:", inputs.shape, features.shape)
        assert inputs.shape[0] == features.shape[0]
        anchor   = features[0:show_x,:]
        positive = features[show_x:2*show_x,:]
        negative = features[2*show_x:3*show_x,:]
        dp = np.sum( np.power((anchor - positive), 2), axis=1 )
        dn = np.sum( np.power((anchor - negative), 2), axis=1)
        print("showBatch:", dp, dn)
    show_sample_img = np.zeros( (show_y*im_width, show_x*im_heigh, 3), dtype=np.uint8)
    x=0
    y=0
    for b_idx in range(inputs.shape[0]):
        im = inputs[b_idx]
        label = labels[b_idx]
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
        x = x+1
        if x==show_x:
            y = y+1
            x = 0
            if y==show_y:
                y = 0
                cv2.imshow("sample", show_sample_img)
                cv2.waitKey()

if __name__=='__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='triplet image iter')
    parser.add_argument('--data-root',  default='/home/ubuntu/zms/data/dl2dl3/')
    parser.add_argument('--batch-size', default=8)
    parser.add_argument('--image-size', default='112, 112')
    parser.add_argument('--model-path', default='/home/ubuntu/zms/models/ResNet_50_Epoch_33.pth')
    args = parser.parse_args()
    im_width = int(args.image_size.split(',')[0])
    im_heigh = int(args.image_size.split(',')[1])
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
    
    dataset_train = TripletHardImgData(os.path.join(args.data_root, 'imgs'), model, transform=train_transform, use_list=False)
    # dataset_train = TripletHardImgData( os.path.join(args.data_root, 'imgs.lst'), model, transform=train_transform, use_list=True)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size = args.batch_size, shuffle=True, sampler = None,
        pin_memory = True, num_workers = 4, drop_last = True
    )
    start = time.time()
    model.eval()
    embeddings = np.zeros([args.batch_size*3, embedding_size])
    for epoch in range(10):
        print("epoch %d"%epoch)
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
            network_in = inputs.to(DEVICE).cpu()
            features = model(network_in)
            features = F.normalize(features).detach()

            anchor   = features[0:args.batch_size,:]
            positive = features[args.batch_size:2*args.batch_size,:]
            negative = features[2*args.batch_size:3*args.batch_size,:]
            d_pos = (anchor - positive).pow(2).sum(1)
            d_neg = (anchor - negative).pow(2).sum(1)
            dp = d_pos.numpy()
            dn = d_neg.numpy()

            print("average p_dist:", np.average(dp))
            print("average n_dist:", np.average(dn))
           
            inputs = inputs.numpy()
            labels = labels.numpy()
            features = features.numpy()
            showBatch(inputs, labels, features, args.batch_size, show_y=3)

    print("10 epoch use time: %.2f s"%(time.time()-start))
