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
import multiprocessing
from multiprocessing import Lock, Process
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
class TripletHardImgData(data.Dataset):
    def __init__(self, root, model, \
                 batch_size, bag_size, input_size, \
                 transform=None, target_transform=None, \
                 n_workers = 4, embedding_size = 512):
        super(TripletHardImgData, self).__init__()
        print("triplet hard image dataloader inited")
        self.transform = transform               # transform datas
        self.target_transform = target_transform # transform targets
        self.embedding_size = embedding_size
        self.model = model
        self.batch_size = batch_size
        self.bag_size = bag_size
        self.input_size = input_size
        self._cur = 0
        self.bag_img_seq = []
        self.bag_lab_seq = []
        self.n_workers = n_workers
        self._inited = False
        samples = self._read_paths(root)
        self.samples = samples                 # TODO: shuffle by batch
        self.targets = [s[1] for s in samples] # targets is a list with ids

    def _read_paths(self, path):
        samples = []
        with open(path, 'r') as fp:
            for line in fp.readlines():
                [path, id] = line.strip().split(' ')
                samples.append( ( path, int(id) ) )    # match torchvison Folder fomat 
        return samples 

    def __getitem__(self, index):
        return self.bag_img_seq[index], self.bag_lab_seq[index]

    def __len__(self):
        if( self._inited ): return len(self.bag_img_seq)
        else: return len(self.samples)
    
    def _load_func(self, qin, pout):  
        # print("load process {}".format(os.getpid()))
        for item in qin:
            path, target = item 
            img = pil_loader(path) 
            # img = cv2.imread(path) 
            pout.append( (img, int(target)) )

    def _get_bag(self):
        start = time.time()
        bag = []
        if (self._cur + self.bag_size) > len(self.samples) :
            self._cur = 0    # not enough for a bag
        bag_samples = self.samples[self._cur:self._cur+self.bag_size]
        self._cur += self.bag_size
        if ( self.n_workers > 1):
            q_in = [[]for i in range(self.n_workers)]
            length = self.bag_size/self.n_workers
            for idx in range(self.bag_size):
                q_in[idx%len(q_in)].append(bag_samples[idx])
            with multiprocessing.Manager() as MG:
                p_tuple = [ multiprocessing.Manager().list() for i in range(self.n_workers) ]
                loaders = [ Process( target =self._load_func, \
                    args=(q_in[i], p_tuple[i] ) ) for i in range(self.n_workers) ]
            print("sep:%.4f"%(time.time()-start), end=' ' )
            
            start = time.time()
            for p in loaders: p.start()
            for p in loaders: p.join() 
            print("load:%.4f"%(time.time()-start), end=' ' )

            start = time.time()
            for imgs in p_tuple:
                bag.extend(imgs)
            print("gather:%.4f"%(time.time()-start) )
   
        else:
            for idx in range(self.bag_size):
                path, target = bag_samples[idx]
                img = pil_loader(path)
                bag.append( (img, int(target)) )
        return bag
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

    def reset(self, model=None, device="cpu"):
        self._inited = True
        start = time.time()
        bagTuples = self._get_bag()
        print("loadImg: %.4f s"%((time.time()-start)), end=' ')    
        
        start = time.time()
        bagImg = torch.empty(self.bag_size, 3, self.input_size[0], self.input_size[1])
        bagLabel = torch.LongTensor(self.bag_size, 1).zero_()
        for i in range(self.bag_size):
           image, bagLabel[i] = bagTuples[i]
           bagImg[i] = self.transform(image)
        print("transTensor: %.4f s"%((time.time()-start)), bagImg.size(),end=' ')
        start = time.time()
        features = torch.empty(self.bag_size, self.embedding_size)
        for b_idx in range(int(self.bag_size/self.batch_size)):
            batchIn = bagImg[ b_idx*self.batch_size : (b_idx+1)*self.batch_size, : ]
            fea = model(batchIn.to(device))
            fea = F.normalize(fea)
            features[ b_idx*self.batch_size : (b_idx+1)*self.batch_size, : ] = fea.detach().cpu()
        print("bagFeature: %.4f s"%((time.time()-start)),end=' ')
        start = time.time()
        dist_matrix = torch.empty(self.bag_size, self.bag_size)
        dist_matrix = self._get_dist(features.numpy())
        baglable_1v = bagLabel.view(self.bag_size).numpy() #longTensor -> numpy = int64
        for a_idx in range( self.bag_size ):
            p_dist = dist_matrix[a_idx].copy()
            n_dist = dist_matrix[a_idx]
            a_lable = baglable_1v[a_idx]
            if(np.sum(baglable_1v==a_lable) == 1):
                p_idx = a_idx
            else:
                p_dist[np.where(baglable_1v!=a_lable)] = 0
                numCandidate = int( max(1, 0.3*np.sum(baglable_1v==a_lable)))
                p_candidate = p_dist.argsort()[-numCandidate:]
                p_idx = np.random.choice(p_candidate)
            n_dist[np.where(baglable_1v==a_lable)] = 0
            numCandidate = int( max(1, 0.1*self.bag_size) )
            n_candidate = n_dist.argsort()[:numCandidate]
            n_idx = np.random.choice(n_candidate)
            self.bag_img_seq.append( (bagImg[a_idx], bagImg[p_idx], bagImg[n_idx]) )
            self.bag_lab_seq.append( (bagLabel[a_idx], bagLabel[p_idx], bagLabel[n_idx]) )
        print("dist: %.4f s"%((time.time()-start)))
        sys.stdout.flush()
if __name__=='__main__':
    print("main process {}".format(os.getpid()))
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='triplet image iter')
    parser.add_argument('--data-root', type=str, default='/home/ubuntu/zms/data/ms1m_emore_img/')
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--image-size', default=[112, 112])
    parser.add_argument('--model-path', type=str, default='/home/ubuntu/zms/models/ResNet_50_Epoch_33.pth')
    parser.add_argument('--bag-size', type=int, default=120)
    parser.add_argument('--load-workers', type=int, default=4)
    args = parser.parse_args()
    im_width = args.image_size[0]
    im_heigh = args.image_size[1]
    print("triplet hard image iter size:", im_width, im_heigh)

    model = ResNet_50([im_width, im_heigh])
    embedding_size = 512
    train_transform = transforms.Compose([ 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean =  [0.5, 0.5, 0.5], std =  [0.5, 0.5, 0.5]),
    ])
    dataset_train = TripletHardImgData( os.path.join(args.data_root, 'imgs.lst'), model, \
                    batch_size = args.batch_size, bag_size = args.bag_size, input_size = [112,112], \
                    transform= train_transform, n_workers = args.load_workers, use_list=True)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size = args.batch_size, shuffle=True, sampler = None,
        pin_memory = True, num_workers = 0, drop_last = True
    )
    model.eval()
    model.to(DEVICE)
    embeddings = np.zeros([args.batch_size*3, embedding_size])
    for epoch in range(600000):
        start = time.time()
        print("epoch %d"%epoch)
        dataset_train.reset(model, DEVICE)
        print("epoch use time: %.2f s"%(time.time()-start))
