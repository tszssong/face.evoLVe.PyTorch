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
    def __init__(self, root, model, \
                 batch_size, bag_size, input_size, \
                 transform=None, target_transform=None, use_list = True):
        super(TripletHardImgData, self).__init__()
        print("triplet hard image dataloader inited")
        self.root = root
        self.transform = transform               # transform datas
        self.target_transform = target_transform # transform targets
        self.model = model
        self.batch_size = batch_size
        self.bag_size = bag_size
        self.input_size = input_size
        self._cur = 0
        self.bag_img_seq = []
        self.bag_lab_seq = []
        if use_list:
            samples = self._read_paths(self.root)
        else:
            classes, class_to_idx = self._find_classes(self.root)
            samples = make_dataset(self.root, class_to_idx, extensions=IMG_EXTENSIONS)
            if len(samples) == 0:
                raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                    "Supported extensions are: " + ",".join(extensions)))
            self.class_to_idx = class_to_idx
            self.classes = classes
        # random.shuffle(samples)
        self.samples = samples                #samples is a list like below:
        start = time.time()
        # random.shuffle(self.samples)
        # #[(path1,id1),(path2,id1),(path3,id1), (path4,id2),(path5,id2).....]
        self.targets = [s[1] for s in samples] # targets is a list with ids
        # print( "shuffle %d samples used time: %.2f s"%(len(self.samples), time.time()-start) )
        sys.stdout.flush()
        self.reset(self.model)

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):   # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _read_paths(self, path):
        images = []
        with open(path, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                [path, id] = line.strip().split(' ')
                images.append( ( path, int(id) ) )    # match torchvison Folder fomat 
        return images 

    def __getitem__(self, index):
        return self.bag_img_seq[index], self.bag_lab_seq[index]

    #https://blog.csdn.net/Tan_HandSome/article/details/82501902
    def _get_dist(self, emb):
        vecProd = np.dot(emb, emb.transpose())
        sqr_emb = emb**2
        sum_sqr_emb = np.matrix( np.sum(sqr_emb, axis=1) )
        ex_sum_sqr_emb = np.tile(sum_sqr_emb.transpose(), (1, vecProd.shape[1]))

        sqr_et = emb**2
        sum_sqr_et = np.sum(sqr_et, axis=1)
        ex_sum_sqr_et = np.tile(sum_sqr_et, (vecProd.shape[0], 1))
        sq_ed = ex_sum_sqr_emb + ex_sum_sqr_et - 2*vecProd
        sq_ed[sq_ed<0] = 0.0
        ed = np.sqrt(sq_ed)

        return np.asarray(ed)
   
    def _get_bag(self):
        bagdata = torch.empty(self.bag_size, 3, self.input_size[0], self.input_size[1])
        baglabel = torch.empty(self.bag_size, 1)
        index = 0
        if self._cur + self.bag_size > len(self.samples):
            self._cur = 0    # not enough for a bag
            # random.shuffle(self.samples)
            # self.targets = [s[1] for s in samples]
        while index<self.bag_size:
            path, target = self.samples[index + self._cur]
            img = pil_loader(path)
            if self.transform is not None:
                img = self.transform(img)
            bagdata[index] = img
            baglabel[index] = target
            index = index + 1
        self._cur += self.bag_size
        
        return bagdata, baglabel

    def reset(self, model=None, device="cpu"):
        model.eval()
        model.to(device)
        start = time.time()
        self.bag_img_seq = []
        self.bag_lab_seq = []
        bagdata, baglabel = self._get_bag()
        # if(np.where(baglabel==baglabel[0])[0].size == baglabel.size(0))
        print("loadImg time: %.6f s"%((time.time()-start)), end=' ')    
        start = time.time()
        # features = torch.empty(self.bag_size, self.embedding_size)
        features = torch.empty(self.bag_size, 512)
        for idx in range(int(self.bag_size/self.batch_size)):
            fea = model(bagdata[idx*self.batch_size:(idx+1)*self.batch_size,:].to(device))
            fea = F.normalize(fea).detach()
            features[ idx*self.batch_size:(idx+1)*self.batch_size,: ] = fea
        features.cpu()
        print("getFeature time: %.6f s"%((time.time()-start))) 
        # features = model( bagdata.to(device) )
        # features = F.normalize(features).detach().cpu()
        start = time.time()
        dist_matrix = torch.empty(self.bag_size, self.bag_size)
        dist_matrix = self._get_dist(features.numpy())
        np.set_printoptions(suppress=True)
        print("dataLoader reset:", bagdata.size(), "distMatric use time: %.6f s"%((time.time()-start)))
        sys.stdout.flush()
        assert dist_matrix.shape[0] == self.bag_size
        baglabel_1v = baglabel.view(baglabel.shape[0]).numpy().astype(np.int32)
        for a_idx in range( self.bag_size ):
            p_dist = dist_matrix[a_idx].copy()
            n_dist = dist_matrix[a_idx]
            a_label = baglabel_1v[a_idx]
            
            # TODO:  skip 1 img/id
            if(np.sum(baglabel_1v==a_label) == 1):
                p_idx = a_idx
            else:
                p_dist[np.where(baglabel_1v!=a_label)] = 0
                numCandidate = int( max(1, 0.5*np.where(baglabel_1v==a_label)[0].shape[0]) )
                p_candidate = p_dist.argsort()[-numCandidate:]
                p_idx = np.random.choice( p_candidate )
            # TODO: incase batch_size < class id images
            n_dist[ np.where(baglabel_1v==a_label) ] = 2048    #fill same ids with a bigNumber
            numCandidate = int( max(1, self.bag_size*0.1) )
            # numCandidate = 1
            n_candidate = n_dist.argsort()[ :numCandidate ]    
            n_idx = np.random.choice(n_candidate)
            # print("dist after:", n_dist)
            # print("triplet find:", a_idx, p_idx, n_idx)
            self.bag_img_seq.append((bagdata[a_idx],bagdata[p_idx],bagdata[n_idx]))
            self.bag_lab_seq.append( (int( baglabel_1v[a_idx] ), \
                                      int( baglabel_1v[p_idx] ), \
                                      int( baglabel_1v[n_idx] ) ) ) 
          

    def __len__(self):
        return len(self.bag_img_seq)
        # return len(self.samples)

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
