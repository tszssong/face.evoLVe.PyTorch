import os, sys, time
import os.path
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
import cv2
import jpeg4py as jpeg
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


class FolderImgData(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(FolderImgData, self).__init__()
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
        self.samples = samples
        self.targets = [s[1] for s in samples]
        print("samples:", len(self.samples))

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        # sample = pil_loader(path)               #pil load imgs
        sample = jpeg4py_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)


if __name__=='__main__':
    RESIZE_SCALE = [1.2, 1.0]
    INPUT_SIZE   = [112, 112]       # support: [112, 112] and [224, 224]
    RGB_MEAN     = [0.5, 0.5, 0.5]  # for normalize inputs to [-1, 1]
    RGB_STD      = [0.5, 0.5, 0.5]
    DATA_ROOT = '/home/ubuntu/zms/data/msceleb'
    show_x = 6
    show_y = 3
    train_transform = transforms.Compose([ 
        transforms.Resize([int(RESIZE_SCALE[0]*INPUT_SIZE[0]), int(RESIZE_SCALE[1]*INPUT_SIZE[0])]), # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN, std = RGB_STD),
    ])
    dataset_train = FolderImgData(os.path.join(DATA_ROOT, 'imgs'), train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size = 2, sampler = None, pin_memory = True,
        num_workers = 4, drop_last = True
    )
    start = time.time()
    show_sample_img = np.zeros((show_y*INPUT_SIZE[1], show_x*INPUT_SIZE[0], 3), dtype=np.uint8)
    x=0
    y=0
    for epoch in range(10):
        print("epoch %d"%epoch)
        for inputs, labels in iter(train_loader):
            inputs = inputs.numpy()
            labels = labels.numpy()
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
                        cv2.waitKey()

    print("10 epoch use time: %.2f s"%(time.time()-start))