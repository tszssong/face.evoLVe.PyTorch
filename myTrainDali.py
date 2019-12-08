import os, sys, time, argparse, socket
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from config import configurations
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_18, IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from backbone.model_resa import RA_92
from backbone.model_m2 import MobileV2
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax, Softmax,Combine
from loss.loss import FocalLoss, TripletLoss
from util.utils import make_weights_for_balanced_classes, get_val_data, get_val_pair, separate_irse_bn_paras, separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy
from tqdm import tqdm
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as dali_ops
import nvidia.dali.types as dali_types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from imgdata.dali_img_iter import reader_pipeline
# class reader_pipeline(Pipeline):
#     def __init__(self, image_dir, batch_size, num_threads, device_id):
#         super(reader_pipeline, self).__init__(batch_size, num_threads, device_id)
#         self.input = dali_ops.FileReader(file_root = image_dir, random_shuffle = True)
#         self.decode = dali_ops.ImageDecoder(device = 'mixed', output_type = dali_types.RGB)
#         self.cmn_img = dali_ops.CropMirrorNormalize(device = "gpu",
#                                            crop=(112, 112),  crop_pos_x=0, crop_pos_y=0,
#                                            output_dtype = dali_types.FLOAT, image_type=dali_types.RGB,
#                                            mean=[0.5*255, 0.5*255, 0.5*255],
#                                            std=[0.5*255, 0.5*255, 0.5*255]
#                                            )
#         self.brightness_change = dali_ops.Uniform(range=(0.6,1.4))
#         self.rd_bright = dali_ops.Brightness(device="gpu")
#         self.contrast_change = dali_ops.Uniform(range=(0.6,1.4))
#         self.rd_contrast = dali_ops.Contrast(device = "gpu")
#         self.saturation_change = dali_ops.Uniform(range=(0.6,1.4))
#         self.rd_saturation = dali_ops.Saturation(device = "gpu")
#         self.jitter_change = dali_ops.Uniform(range=(1,2))
#         self.rd_jitter = dali_ops.Jitter(device = "gpu")
#         self.jitter_mask = dali_ops.CoinFlip(probability = 0.3)
#         self.hue_change = dali_ops.Uniform(range = (-30,30))
#         self.hue = dali_ops.Hue(device = "gpu")
#         self.p_hflip = dali_ops.CoinFlip(probability = 0.5)
#         self.flip = dali_ops.Flip(device = "gpu")

#     def define_graph(self):
#         jpegs, labels = self.input(name="Reader")
#         images = self.decode(jpegs)
#         brightness = self.brightness_change()
#         images = self.rd_bright(images, brightness=brightness)
#         contrast = self.contrast_change()
#         images = self.rd_contrast(images, contrast = contrast)
#         saturation = self.saturation_change()
#         images = self.rd_saturation(images, saturation = saturation)
#         jitter = self.jitter_change()
#         jitter_mask = self.jitter_mask()
#         images = self.rd_jitter(images, mask = jitter_mask)
#         hue = self.hue_change()
#         images = self.hue(images, hue = hue)
#         p_hflip = self.p_hflip()
#         images = self.flip(images, horizontal = p_hflip)
#         imgs = self.cmn_img(images)
#         return (imgs, labels)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--data-root', type=str, default='/cloud_data01/zhengmeisong/data/gl2ms1m_img/')
    parser.add_argument('--model-root', type=str, default='../py-model')
    parser.add_argument('--backbone-resume-root', type=str, default='./home/ubuntu/zms/models/ResNet_50_Epoch_33.pth')
    parser.add_argument('--backbone-name', type=str, default='MobileV2') # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    parser.add_argument('--head-resume-root', type=str, default='./home/ubuntu/zms/models/ResNet_50_Epoch_33.pth')
    parser.add_argument('--head-name', type=str, default='Combine') # support: ['Combin', 'Softmax', 'ArcFace', 'CosFace']
    parser.add_argument('--input-size', type=str, default="112, 112")
    parser.add_argument('--loss-name', type=str, default='Focal')  # support: ['FocalLoss', 'Softmax']
    parser.add_argument('--emb-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr-stages', type=str, default="9,15,20")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num-epoch', type=int, default=25)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--disp-freq', type=int, default=1)
    parser.add_argument('--num-classes', type=int, default=143474)
    args = parser.parse_args()

    #======= hyperparameters & data loaders =======#
    torch.manual_seed(args.seed)  # random seed for reproduce results
    LOSS_NAME = args.loss_name # support: ['Focal', 'Softmax']

    INPUT_SIZE = [ int(args.input_size.split(',')[0]), int(args.input_size.split(',')[1]) ]
    lrStages = [int(i) for i in args.lr_stages.strip().split(',')]
   
    GPU_ID = [int(i) for i in args.gpu_ids.split(",") ]
    DEVICE =  torch.device("cuda:%d"%GPU_ID[0] if torch.cuda.is_available() else "cpu")
    print(DEVICE, GPU_ID)
    MULTI_GPU = ( len(GPU_ID)>1 ) # flag to use multiple GPUs

    print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    print("=" * 60, "\nOverall Configurations:\n", args)
    sys.stdout.flush()

    train_dir = os.path.join(args.data_root, 'data_100')
    train_pipes = reader_pipeline(train_dir, args.batch_size, args.num_workers, device_id = GPU_ID[0])
    train_pipes.build()
    train_loader = DALIGenericIterator(train_pipes, ['imgs', 'labels'],\
                                       train_pipes.epoch_size("Reader"), \
                                       auto_reset=True)
    NUM_CLASS = args.num_classes
    NUM_CLASS = 100
    # lfw, cfp_fp, agedb, lfw_issame, cfp_fp_issame, agedb_issame = get_val_data(DATA_ROOT)
    lfw, lfw_issame = get_val_pair(args.data_root, 'lfw')
    cfp_fp, cfp_fp_issame = get_val_pair(args.data_root, 'cfp_fp')
    agedb, agedb_issame = get_val_pair(args.data_root, 'agedb_30')
    BACKBONE = eval(args.backbone_name)(input_size = INPUT_SIZE, emb_size = args.emb_size)
    HEAD = eval(args.head_name)(in_features = args.emb_size, out_features = NUM_CLASS, device_id = GPU_ID)
   
    LOSS_DICT = {'Focal': FocalLoss(), 
                 'Triplet': TripletLoss(), 
                 'Softmax': nn.CrossEntropyLoss()}
    LOSS = LOSS_DICT[LOSS_NAME]
    
    if args.backbone_name.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_resnet_bn_paras(HEAD)
    OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 
                            'weight_decay': args.weight_decay}, 
                            {'params': backbone_paras_only_bn}], lr = args.lr, momentum = args.momentum)
    print(LOSS,"\n",OPTIMIZER,"\n","="*60, "\n") 
    sys.stdout.flush() 
    # optionally resume from a checkpoint
    if args.backbone_resume_root:
        print("=" * 60)
        if os.path.isfile(args.backbone_resume_root):
            print("Loading Backbone Checkpoint '{}'".format(args.backbone_resume_root))
            BACKBONE.load_state_dict(torch.load(args.backbone_resume_root,map_location=DEVICE))
        else:
            print("No Checkpoint Found at '{}'.".format(args.backbone_resume_root))
    if args.head_resume_root:   
        if os.path.isfile(args.head_resume_root):
            print("Loading Head Checkpoint '{}'".format(args.head_resume_root))
            HEAD.load_state_dict(torch.load(args.head_resume_root,map_location=DEVICE))
        else:
            print("No Checkpoint Found at '{}'. ".format(args.head_resume_root))
        print("=" * 60)
        sys.stdout.flush() 
    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)

    #======= train & validation & save checkpoint =======#
    DISP_FREQ = args.disp_freq                      # frequency to display training loss & acc
    NUM_EPOCH_WARM_UP = args.num_epoch // 25  # use the first 1/25 epochs to warm up
    NUM_BATCH_WARM_UP = 0  # use the first 1/25 epochs to warm up
    batch = 0  # batch index
    elasped = 0.0
    for epoch in range(args.num_epoch): # start training process
        for l_idx in range(len(lrStages)):
                if epoch == lrStages[l_idx]:
                    schedule_lr(OPTIMIZER)

        BACKBONE.train()  # set to training mode
        HEAD.train()

        losses = AverageMeter()
        top1   = AverageMeter()
        top5   = AverageMeter()

        # for inputs, labels in tqdm(iter(train_loader)):
        for i, datas  in enumerate(train_loader):
            inputs = datas[0]['imgs']
            labels = datas[0]['labels']
            labels = labels.view(args.batch_size)
            
            start = time.time()
            if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (batch + 1 <= NUM_BATCH_WARM_UP):  # adjust LR during warm up
                warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, args.lr, OPTIMIZER)
            # compute output
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()
            features = BACKBONE(inputs)
            outputs = HEAD(features, labels)
            loss = LOSS(outputs, labels)
            print('loss:',loss.cpu().detach().numpy())
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels, topk = (1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))
            top5.update(prec5.data.item(), inputs.size(0))
            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()
            end = time.time()
            elasped = elasped + (end - start)
            # dispaly training loss & acc every DISP_FREQ
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                #print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "average:%.2f s/batch"%(end-start) )
                print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%.3f/batch"%((end-start)) )
                elasped = 0
                print('Epoch {}/{} Batch {}\t'
                      'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch + 1, args.num_epoch, batch + 1, loss = losses, top1 = top1, top5 = top5))
                print("=" * 60)
                sys.stdout.flush()
            batch += 1 # batch index

        # training statistics per epoch (buffer for visualization)
        epoch_loss = losses.avg
        epoch_acc = top1.avg
        print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
        print('Epoch: {}/{}\t'
              'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            epoch + 1, args.num_epoch, loss = losses, top1 = top1, top5 = top5))
        print("=" * 60)
        sys.stdout.flush() 

        # perform validation & save checkpoints per epoch
        # # validation statistics per epoch (buffer for visualization)
        # print("=" * 60)
        # print("Perform Evaluation on LFW, CFP_FF, CFP_FP, AgeDB, CALFW, CPLFW and VGG2_FP, and Save Checkpoints...")
        # accuracy_lfw, best_threshold_lfw = perform_val(MULTI_GPU, DEVICE, args.emb_size, args.batch_size, BACKBONE, lfw, lfw_issame)
        # accuracy_cfp_fp, best_threshold_cfp_fp = perform_val(MULTI_GPU, DEVICE, args.emb_size, args.batch_size, BACKBONE, cfp_fp, cfp_fp_issame)
        # accuracy_agedb, best_threshold_agedb = perform_val(MULTI_GPU, DEVICE, args.emb_size, args.batch_size, BACKBONE, agedb, agedb_issame)
        # print("Epoch {}/{}, Evaluation: LFW Acc: {}, CFP_FP Acc: {}, AgeDB Acc: {}".format(epoch + 1, args.num_epoch, accuracy_lfw, accuracy_cfp_fp, accuracy_agedb))
        # print("=" * 60)
        # sys.stdout.flush() 
        # save checkpoints per epoch
        if MULTI_GPU:
            torch.save(BACKBONE.module.state_dict(), os.path.join(args.model_root, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(args.backbone_name, epoch + 1, batch, get_time())))
            torch.save(HEAD.state_dict(), os.path.join(args.model_root, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(args.head_name, epoch + 1, batch, get_time())))
        else:
            torch.save(BACKBONE.state_dict(), os.path.join(args.model_root, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(args.backbone_name, epoch + 1, batch, get_time())))
            torch.save(HEAD.state_dict(), os.path.join(args.model_root, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(args.head_name, epoch + 1, batch, get_time())))
