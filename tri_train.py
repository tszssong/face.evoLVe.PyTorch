import os, sys, time, os.path, argparse, socket
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
sys.path.append( os.path.join( os.path.dirname(__file__),'/imgdata/') )
from config import configurations
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from loss.loss import FocalLoss, TripletLoss
from util.utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy
from tensorboardX import SummaryWriter
from tqdm import tqdm
from imgdata.tri_img_iter import TripletImgData
from imgdata.tri_hard_iter import TripletHardImgData
from imgdata.show_img import showBatch
hostname = socket.gethostname()
torch.manual_seed(1337)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='/home/ubuntu/zms/data/ms1m_emore100')
    parser.add_argument('--model-root', type=str, default='../py-model')
    parser.add_argument('--log-root', type=str, default='../py-log')
    parser.add_argument('--backbone-resume-root', type=str, default='/home/ubuntu/zms/models/ResNet_50_Epoch_33.pth')
    parser.add_argument('--head-resume-root', type=str, default='')
    parser.add_argument('--backbone-name', type=str, default='ResNet_50') # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    parser.add_argument('--input-size', type=str, default="112, 112")
    parser.add_argument('--loss-name', type=str, default='TripletLoss')  # support: ['FocalLoss', 'Softmax', 'TripletLoss']
    parser.add_argument('--embedding-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--bag-size', type=int, default=512)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--lr-stages', type=str, default="120000, 165000, 195000")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num-epoch', type=int, default=100000)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--disp_freq', type=int, default=1)
    parser.add_argument('--test_epoch', type=int, default=2000)
    args = parser.parse_args()
    writer = SummaryWriter(args.log_root) # writer for buffering intermedium results
    margin = args.margin

    INPUT_SIZE = [ int(args.input_size.split(',')[0]), int(args.input_size.split(',')[1]) ]
    lrStages = [int(i) for i in args.lr_stages.strip().split(',')]
   
    DEVICE =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    GPU_ID = [int(i) for i in args.gpu_ids.split(",") ]
    print(DEVICE, GPU_ID)
    MULTI_GPU = ( len(GPU_ID)>1 ) # flag to use multiple GPUs
  
    print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    print("=" * 60, "\nOverall Configurations:\n", args)
    
    BACKBONE = eval(args.backbone_name)(INPUT_SIZE)
    print("=" * 60, "\n", BACKBONE, "\n{} Backbone Generated".format(args.backbone_name),"\n","="*60)
    
    if args.backbone_name.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
     # optionally resume from a checkpoint
    if args.backbone_resume_root and os.path.isfile(args.backbone_resume_root):
        print("Loading Backbone Checkpoint '{}'".format(args.backbone_resume_root))
        BACKBONE.load_state_dict(torch.load(args.backbone_resume_root)) 
    else:
        print("No Checkpoint, Train from Scratch" )
    if args.loss_name == 'Softmax':
        LOSS = nn.CrossEntropyLoss()
    else:
        LOSS = eval(args.loss_name)(args.margin)

    OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn, 'weight_decay': args.weight_decay}, \
                           {'params': backbone_paras_only_bn}], lr = args.lr, momentum = args.momentum)
    
    print(LOSS,"\n",OPTIMIZER,"\n","="*60, "\n") 
    sys.stdout.flush() 

    lfw, cfp_ff, cfp_fp, agedb, calfw, cplfw, vgg2_fp, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_issame, calfw_issame, cplfw_issame, vgg2_fp_issame = get_val_data(args.data_root)

    train_transform = transforms.Compose([ 
        transforms.Resize([128, 128]),     # smaller side resized
        transforms.RandomCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean =  [0.5, 0.5, 0.5], std =  [0.5, 0.5, 0.5]),
    ])

    dataset_train = TripletHardImgData( os.path.join(args.data_root, 'imgs.lst'), BACKBONE, \
        batch_size = args.batch_size, bag_size = args.bag_size, input_size = INPUT_SIZE,\
        transform=train_transform, use_list=True)
        
    # dataset_train = TripletHardImgData(os.path.join(args.data_root, 'imgs'), model, transform=train_transform, use_list=False)
    train_loader = torch.utils.data.DataLoader( dataset_train, batch_size = args.batch_size, shuffle=True, \
                                  pin_memory = True, num_workers = args.num_workers, drop_last = True )

    # NUM_CLASS = len(train_loader.dataset.classes)
    # print("Number of Training Classes: {}".format(NUM_CLASS))
    sys.stdout.flush()  
    if MULTI_GPU:   # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:           # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)
    
    #======= train & validation & save checkpoint =======#
    batch = 0   # batch index
    margin_count = 0
    for epoch in range(args.num_epoch): # start training process
        for l_idx in range(len(lrStages)):
            if epoch == lrStages[l_idx]:
                schedule_lr(OPTIMIZER)
       
        losses = AverageMeter()
        top1   = AverageMeter()

        # for inputs, labels in tqdm(iter(train_loader)):
        dataset_train.reset(BACKBONE,DEVICE)
        start = time.time()
        BACKBONE.train()  # set to training mode
        for inputs, labels in iter(train_loader):
            a = inputs[0]
            p = inputs[1]
            n = inputs[2]
            inputs = torch.cat((a,p,n), 0) 
            a_label = labels[0]
            p_label = labels[1]
            n_label = labels[2]
            labels = torch.cat((a_label, p_label, n_label), 0)
            # compute output
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()
            outputs = BACKBONE(inputs)
            # show batch data: only on ubuntu
            if hostname=="ubuntu-System-Product-Name":
                features = F.normalize(outputs).detach()
                showBatch(inputs.cpu().numpy(), labels.cpu().numpy(), features.cpu().numpy(), args.batch_size)
            
            loss, loss_batch = LOSS(outputs, labels, margin)
            loss_batch = loss_batch.detach().cpu().numpy()
            n_err = np.where(loss_batch!=0)[0].shape[0] 
            prec = 1.0 - float(n_err) / loss_batch.shape[0]
            
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec, inputs.size(0))
            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()
            batch += 1 # batch index
            
        # training statistics per epoch (buffer for visualization)
        # if(top1.avg > 0.95):
        #     margin_count += 1
        # elif(top1.avg < 0.85)and (margin>0.1):
        #     margin_count -= 1
            
        # if margin_count == 5:
        #     margin += 0.01
        #     print("margin fixed to:", margin, "margin count:%d"%margin_count)
        #     sys.stdout.flush()
        #     margin_count = 0
        # elif margin_count == -5 and margin>0.1:
        #     margin -= 0.01
        #     print("margin fixed to:", margin, "margin count:%d"%margin_count)
        #     sys.stdout.flush()
        #     margin_count = 0

        epoch_loss = losses.avg
        epoch_acc = top1.avg
        writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
        writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
        print( time.strftime("%Y-%m-%d %H:%M:%S\t", time.localtime()), "%.3f s/epoch"%(time.time()-start) )
        print('Epoch: {}/{}\t'
            'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Training Prec {top1.val:.3f} ({top1.avg:.3f})'.format(
            epoch + 1, args.num_epoch, loss = losses,top1 = top1))
        print("=" * 60)
        sys.stdout.flush() 

        if (epoch%args.test_epoch==0 and epoch!=0):
            # perform validation & save checkpoints per epoch
            print("=" * 60, "\nPerform Evaluation on LFW, CFP_FF, CFP_FP, AgeDB, CALFW, CPLFW and VGG2_FP, and Save Checkpoints...")
            sys.stdout.flush() 
            accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, args.embedding_size, args.batch_size, BACKBONE, lfw, lfw_issame)
            buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, epoch + 1)
            accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(MULTI_GPU, DEVICE, args.embedding_size, args.batch_size, BACKBONE, cfp_ff, cfp_ff_issame)
            buffer_val(writer, "CFP_FF", accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff, epoch + 1)
            accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(MULTI_GPU, DEVICE, args.embedding_size, args.batch_size, BACKBONE, cfp_fp, cfp_fp_issame)
            buffer_val(writer, "CFP_FP", accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp, epoch + 1)
            accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(MULTI_GPU, DEVICE, args.embedding_size, args.batch_size, BACKBONE, agedb, agedb_issame)
            buffer_val(writer, "AgeDB", accuracy_agedb, best_threshold_agedb, roc_curve_agedb, epoch + 1)
            accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(MULTI_GPU, DEVICE, args.embedding_size, args.batch_size, BACKBONE, calfw, calfw_issame)
            buffer_val(writer, "CALFW", accuracy_calfw, best_threshold_calfw, roc_curve_calfw, epoch + 1)
            accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(MULTI_GPU, DEVICE, args.embedding_size, args.batch_size, BACKBONE, cplfw, cplfw_issame)
            buffer_val(writer, "CPLFW", accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw, epoch + 1)
            accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val(MULTI_GPU, DEVICE, args.embedding_size, args.batch_size, BACKBONE, vgg2_fp, vgg2_fp_issame)
            buffer_val(writer, "VGGFace2_FP", accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp, epoch + 1)
            print("Epoch {}/{}, Evaluation: LFW Acc: {}, CFP_FF Acc: {}, CFP_FP Acc: {}, AgeDB Acc: {}, CALFW Acc: {}, CPLFW Acc: {}, VGG2_FP Acc: {}".format(epoch + 1, args.num_epoch, accuracy_lfw, accuracy_cfp_ff, accuracy_cfp_fp, accuracy_agedb, accuracy_calfw, accuracy_cplfw, accuracy_vgg2_fp))
            print("=" * 60)
            sys.stdout.flush() 

            # save checkpoints per epoch
            if MULTI_GPU:
                torch.save(BACKBONE.module.state_dict(), os.path.join(args.model_root, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(args.backbone_name, epoch + 1, batch, get_time())))
            else:
                torch.save(BACKBONE.state_dict(), os.path.join(args.model_root, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(args.backbone_name, epoch + 1, batch, get_time())))
       