import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import os,sys,cv2,pickle

sys.path.append( os.path.dirname(__file__) )
from verification import evaluate

from datetime import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from PIL import Image
import bcolz
import io

def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def make_weights_for_balanced_classes(images, nclasses):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    return weight


def get_val_pair(path, name):
    if not "pkl" in name:
        carray = bcolz.carray(rootdir = os.path.join(path, name), mode = 'r')
        issame = np.load('{}/{}_list.npy'.format(path, name))
    else:
        loadfile = open(path+'/'+name, 'rb')
        carray,issame = pickle.load(loadfile)
        carray = carray.astype(np.float32)
        issame = issame.astype(np.int64)
        for idx in range(carray.shape[0]):
            carray[idx] = ( carray[idx]-np.mean(carray[idx]) ) / ( np.max(carray[idx])-np.min(carray[idx])+2**-30 )

    print(name,':', len(carray), len(issame))
    return carray, issame

def get_val_pair_img(path, name):
    carray = bcolz.carray(rootdir = os.path.join(path, name), mode = 'r')
    issame = np.load('{}/{}_list.npy'.format(path, name))
    print(type(carray), type(issame))
    print(len(carray), len(issame))
    print( type(issame[0]), issame[0] )
    # t_carry = torch.tensor(carray)
    # print(type(t_carry))
    np_carry = np.array(carray)
    print(type(np_carry),np_carry.shape)
    for idx in range(np_carry.shape[0]):
        im = np_carry[idx]
        # print(im.shape)
        im = im*127.5 + 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1,2,0))
        # im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imshow("im", im)
        cv2.waitKey()
    return carray, issame

def get_val_pair_pickl(path, name):
    loadfile = open(path+'/'+name, 'rb')
    carray,issame = pickle.load(loadfile)
    carray = carray.astype(np.float32)
    issame = issame.astype(np.int64)
    for idx in range(carray.shape[0]):
        carray[idx] = ( carray[idx]-np.mean(carray[idx]) ) / ( np.max(carray[idx])-np.min(carray[idx])+0.00001 )

    print(type(carray), type(issame))
    print(len(carray), len(issame))
    print( type(issame[0]), issame[0] )
   
    np_carry = np.array(carray)
    print(type(np_carry),np_carry.shape)
    for idx in range(np_carry.shape[0]):
        im = np_carry[idx]
        print(issame[int(idx/2)])
        im = im*127.5 + 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1,2,0))
        # im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imshow("im", im)
        cv2.waitKey()
    return carray, issame

tensor_transform = transforms.Compose([  transforms.ToTensor(),
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5] ),
])

def get_jaivs_var_lists(path):
    alist = []
    plist = []
    nlist = []
    for i in range(1572000, 1576000):
        imgname = str(i) + '.jpg'
        try:                             #Do not need to convert rgb, test do it [2,1,0]
            idImg = Image.open(path + '/ids/JPEGImages/' + imgname)     #.convert('RGB')
            spotImg = Image.open(path + '/spots/JPEGImages/' + imgname) 
        except:
            print(imgname, "not exits, skip!!!")
            continue

        if i<1575000:
            negidx = i+1000
        else:
            negidx = i-3000
            
        negname = str(negidx) + '.jpg'
        isValid = False
        offset = 0
        while not isValid and offset<1000:
            offset += 2
            try:
                negImg = Image.open(path + '/spots/JPEGImages/' + negname)
                isValid = True
            except:
                print("Neg",negname, "not exists, one more try...")
                negname = str(negidx+offset) + '.jpg'
                isValid = False
        alist.append(path + '/ids/JPEGImages/' + imgname)
        plist.append(path + '/spots/JPEGImages/' + imgname)
        nlist.append(path + '/spots/JPEGImages/' + negname)
    return alist, plist, nlist

def gen_jaivs_var_data(path, name):
    alist, plist, nlist = get_jaivs_var_lists(path)
    assert(len(alist)==len(plist))
    array = np.zeros([4*len(alist), 3, 112, 112]).astype(np.uint8)
    issame = np.zeros(2*len(alist)).astype(np.uint8)
    for idx in range(len(alist)):
        a = cv2.imread(alist[idx])
        p = cv2.imread(plist[idx])
        n = cv2.imread(nlist[idx])

        a=a.transpose((2,0,1))
        p=p.transpose((2,0,1))
        n=n.transpose((2,0,1))

        array[4*idx] = a
        array[4*idx+1] = p
        array[4*idx+2] = a
        array[4*idx+3] = n
        
        issame[2*idx] = 1
        issame[2*idx+1] = 0

    print(len(issame), len(plist))
    outputfile = open(path+'/'+name+'.pkl', 'wb')
    pickle.dump((array,issame),outputfile)
    outputfile.close()

def gen_valHomo_data(path, name):
    with open(path + '/test.lst') as fp:
        lines = fp.readlines()
        maxId = int(lines[-1].strip().split(' ')[-1]) # 2600
        print("generating test pkl with %d ids"%(maxId))
        array = np.zeros([4*maxId, 3, 112, 112]).astype(np.uint8)
        issame = np.zeros(2*maxId).astype(np.uint8)
        lastLabel = 100
        data_idx = 0
        for idx in range(len(lines)):
            a_path, a_label = lines[idx].strip().split(' ')
            if lastLabel == a_label:
                continue
            lastLabel = a_label
            p_path, p_label = lines[idx+1].strip().split(' ')
            if not a_label==p_label:
                continue
            if idx + 100 < len(lines):
                n_path, n_label = lines[idx+100].strip().split(' ')
            else:
                n_path, n_label = lines[idx-5000].strip().split(' ')
            if a_label==n_label:
                continue
            
            a = cv2.imread(a_path)
            p = cv2.imread(p_path)
            n = cv2.imread(n_path)
            
            a=a.transpose((2,0,1))
            p=p.transpose((2,0,1))
            n=n.transpose((2,0,1))
            
            array[4*data_idx] = a
            array[4*data_idx+1] = p
            array[4*data_idx+2] = a
            array[4*data_idx+3] = n
            
            issame[2*data_idx] = 1
            issame[2*data_idx+1] = 0
            data_idx += 1
            print("processed id:", int(lastLabel))
            if data_idx >= maxId:
                break

    print(len(issame), len(array))
    outputfile = open(path+'/'+name, 'wb')
    pickle.dump((array,issame),outputfile)
    outputfile.close()
         
def get_val_data(data_path):
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    cfp_ff, cfp_ff_issame = get_val_pair(data_path, 'cfp_ff')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    calfw, calfw_issame = get_val_pair(data_path, 'calfw')
    cplfw, cplfw_issame = get_val_pair(data_path, 'cplfw')
    vgg2_fp, vgg2_fp_issame = get_val_pair(data_path, 'vgg2_fp')

    return lfw, cfp_ff, cfp_fp, agedb_30, calfw, cplfw, vgg2_fp, \
           lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_30_issame, \
           calfw_issame, cplfw_issame, vgg2_fp_issame

def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def separate_resnet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)
            
    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))
    
    return paras_only_bn, paras_wo_bn


def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = batch * init_lr / num_batch_warm_up

    print(optimizer)

def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.

    print(optimizer)


def de_preprocess(tensor):

    return tensor * 0.5 + 0.5


hflip = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.functional.hflip,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)

    return hfliped_imgs


ccrop = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.Resize([128, 128]),  # smaller side resized
            transforms.CenterCrop([112, 112]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


def ccrop_batch(imgs_tensor):
    ccropped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        ccropped_imgs[i] = ccrop(img_ten)

    return ccropped_imgs


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize = 14)
    plt.ylabel("TPR", fontsize = 14)
    plt.title("ROC Curve", fontsize = 14)
    plot = plt.plot(fpr, tpr, linewidth = 2)
    buf = io.BytesIO()
    plt.savefig(buf, format = 'jpeg')
    buf.seek(0)
    plt.close()

    return buf


def perform_val(multi_gpu, device, embedding_size, batch_size, backbone, \
                carray, issame, nrof_folds = 10, tta = True):
    if multi_gpu:
        backbone = backbone.module    # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode

    idx = 0
    
    embeddings = np.zeros([len(carray), embedding_size])
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + batch_size][:, [2, 1, 0], :, :])
            if tta:
                ccropped = ccrop_batch(batch)
                fliped = hflip_batch(ccropped)
                emb_batch = backbone(ccropped.to(device)).cpu() + backbone(fliped.to(device)).cpu()
                embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch(batch)
                embeddings[idx:idx + batch_size] = l2_norm(backbone(ccropped.to(device))).cpu()
            idx += batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            if tta:
                ccropped = ccrop_batch(batch)
                fliped = hflip_batch(ccropped)
                emb_batch = backbone(ccropped.to(device)).cpu() + backbone(fliped.to(device)).cpu()
                embeddings[idx:] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch(batch)
                embeddings[idx:] = l2_norm(backbone(ccropped.to(device))).cpu()

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    
    return accuracy.mean(), best_thresholds.mean()


def buffer_val(writer, db_name, acc, best_threshold,  epoch):
    writer.add_scalar('{}_Accuracy'.format(db_name), acc, epoch)
    writer.add_scalar('{}_Best_Threshold'.format(db_name), best_threshold, epoch)
    # writer.add_image('{}_ROC_Curve'.format(db_name), roc_curve_tensor, epoch)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

if __name__ == '__main__':
    print("utils")
    gen_valHomo_data('/home/ubuntu/zms/data/jr-pairs-2w', 'jr7.pkl')
    get_val_pair_pickl('/home/ubuntu/zms/data/jr-pairs-2w','jr7.pkl')
    # gen_valHomo_data('/home/ubuntu/zms/data/ms1m_emore_img', 'gl2ms1mdl23f1ww1.pkl')
    # get_val_pair_pickl('/home/ubuntu/zms/data/ms1m_emore_img','gl2ms1mdl23f1ww1.pkl')
    # gen_jaivs_var_data('/home/ubuntu/zms/data/ja-ivs-test3','ja_ivs')
    # get_val_pair_pickl('/home/ubuntu/zms/data/ja-ivs-test3','ja_ivs.pkl')
    # get_val_pair_img('/home/ubuntu/zms/data/ms1m_emore100/', 'vgg2_fp')
    # get_val_pair_img('/home/ubuntu/zms/data/ms1m_emore100/', 'agedb_30')