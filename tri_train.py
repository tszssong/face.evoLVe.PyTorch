import os, sys, time, os.path, argparse
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
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/home/ubuntu/zms/data/ms1m_emore100')
    parser.add_argument('--model-root', default='../py-model')
    parser.add_argument('--log-root',   default='../py-log')
    parser.add_argument('--backbone-resume-root', default='/home/ubuntu/zms/models/ResNet_50_Epoch_33.pth')
    parser.add_argument('--head-resume-root', default='')
    parser.add_argument('--backbone-name', default='ResNet_50') # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    parser.add_argument('--input-size', default=[112, 112])
    parser.add_argument('--loss-name', default='TripletLoss')  # support: ['FocalLoss', 'Softmax', 'TripletLoss']
    parser.add_argument('--embedding-size', default=512)
    parser.add_argument('--batch-size', default=12)
    parser.add_argument('--bag-size', default=120)
    parser.add_argument('--lr', default=0.05)
    parser.add_argument('--lr-stages', default=[35, 65, 95])
    parser.add_argument('--weight-decay', default=5e-4)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--num-epoch', default=125)
    parser.add_argument('--num-workers', default=6)
    parser.add_argument('--gpu-id', default=[0])
    parser.add_argument('--disp_freq', default=1)
    parser.add_argument('--test_epoch', default=10)
    args = parser.parse_args()
    torch.manual_seed(1337)
    writer = SummaryWriter(args.log_root) # writer for buffering intermedium results

    INPUT_SIZE = args.input_size
    BATCH_SIZE = args.batch_size
    LR = args.lr    # initial LR
    NUM_EPOCH = args.num_epoch
    
    DEVICE =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    GPU_ID = args.gpu_id
    MULTI_GPU = ( len(GPU_ID)>1 ) # flag to use multiple GPUs
  
    print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    print("=" * 60, "\nOverall Configurations:")
    print(args)
    # BACKBONE = eval(args.backbone_name)(args.input_size)
    BACKBONE = ResNet_50([112, 112])
    print("=" * 60)
    print(BACKBONE)
    print("{} Backbone Generated".format(args.backbone_name))
    if args.backbone_name.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
     # optionally resume from a checkpoint
    if args.backbone_resume_root:
        print("=" * 60)
        if os.path.isfile(args.backbone_resume_root):
            print("Loading Backbone Checkpoint '{}'".format(args.backbone_resume_root))
            BACKBONE.load_state_dict(torch.load(args.backbone_resume_root)) 
        else:
            print("No Checkpoint Found at '{}''. Train from Scratch".format(args.backbone_resume_root))
    print("=" * 60)
    sys.stdout.flush() 
    if args.loss_name == 'Softmax':
        LOSS = nn.CrossEntropyLoss()
    else:
        LOSS = eval(args.loss_name)()

    OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn, 'weight_decay': args.weight_decay}, {'params': backbone_paras_only_bn}], lr = LR, momentum = args.momentum)
    
    print(LOSS) 
    print(OPTIMIZER)
    sys.stdout.flush() 

    lfw, cfp_ff, cfp_fp, agedb, calfw, cplfw, vgg2_fp, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_issame, calfw_issame, cplfw_issame, vgg2_fp_issame = get_val_data(args.data_root)

    train_transform = transforms.Compose([ 
        transforms.Resize([128, 128]), # smaller side resized
        transforms.RandomCrop([112, 112]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean =  [0.5, 0.5, 0.5], std =  [0.5, 0.5, 0.5]),
    ])

    dataset_train = TripletHardImgData( os.path.join(args.data_root, 'imgs.lst'), BACKBONE, \
        batch_size = args.batch_size, bag_size = args.bag_size, input_size = [112,112],\
        transform=train_transform, use_list=True)
        
    # dataset_train = TripletHardImgData(os.path.join(args.data_root, 'imgs'), model, transform=train_transform, use_list=False)

    train_loader = torch.utils.data.DataLoader( dataset_train, batch_size = BATCH_SIZE, shuffle=True, \
                                  pin_memory = True, num_workers = args.num_workers, drop_last = True )

    NUM_CLASS = len(train_loader.dataset.classes)
    print("Number of Training Classes: {}".format(NUM_CLASS))
    sys.stdout.flush()  
    if MULTI_GPU:   # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:           # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)
    
    #======= train & validation & save checkpoint =======#
    batch = 0  # batch index
    elasped = 0
    for epoch in range(NUM_EPOCH): # start training process
        for l_idx in range(len(args.lr_stages)):
            if epoch == args.lr_stages[l_idx]:
                schedule_lr(OPTIMIZER)
       
        losses = AverageMeter()
        top1   = AverageMeter()
        top5   = AverageMeter()

        # for inputs, labels in tqdm(iter(train_loader)):
        dataset_train.reset(BACKBONE,DEVICE)
        BACKBONE.train()  # set to training mode
        for inputs, labels in iter(train_loader):
            start = time.time()
            
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
            features = F.normalize(outputs).detach()
            # showBatch(inputs.cpu().numpy(), labels.cpu().numpy(), features.cpu().numpy(), args.batch_size)
            
            loss = LOSS(outputs, labels)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels, topk = (1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))
            top5.update(prec5.data.item(), inputs.size(0))
            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()
            # dispaly training loss & acc every DISP_FREQ
            if ((batch + 1) % args.disp_freq == 0) and batch != 0:
                print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "average:%.2f s/batch"%(elasped/args.disp_freq) )
                elasped = 0
                print('Epoch {}/{} Batch {}/{}\t'
                      'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch + 1, NUM_EPOCH, batch + 1, len(train_loader) * NUM_EPOCH, loss = losses, top1 = top1, top5 = top5))
                print("=" * 60)
                sys.stdout.flush()

            batch += 1 # batch index
            end = time.time() - start
            elasped = elasped + end
        # training statistics per epoch (buffer for visualization)
        epoch_loss = losses.avg
        epoch_acc = top1.avg
        writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
        writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
        print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
        print('Epoch: {}/{}\t'
            'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            epoch + 1, NUM_EPOCH, loss = losses, top1 = top1, top5 = top5))
        print("=" * 60)
        sys.stdout.flush() 

        if (epoch==args.test_epoch):
            # perform validation & save checkpoints per epoch
            print("=" * 60)
            print("Perform Evaluation on LFW, CFP_FF, CFP_FP, AgeDB, CALFW, CPLFW and VGG2_FP, and Save Checkpoints...")
            accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, args.embedding_size, BATCH_SIZE, BACKBONE, lfw, lfw_issame)
            buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, epoch + 1)
            accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(MULTI_GPU, DEVICE, args.embedding_size, BATCH_SIZE, BACKBONE, cfp_ff, cfp_ff_issame)
            buffer_val(writer, "CFP_FF", accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff, epoch + 1)
            accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(MULTI_GPU, DEVICE, args.embedding_size, BATCH_SIZE, BACKBONE, cfp_fp, cfp_fp_issame)
            buffer_val(writer, "CFP_FP", accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp, epoch + 1)
            accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(MULTI_GPU, DEVICE, args.embedding_size, BATCH_SIZE, BACKBONE, agedb, agedb_issame)
            buffer_val(writer, "AgeDB", accuracy_agedb, best_threshold_agedb, roc_curve_agedb, epoch + 1)
            accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(MULTI_GPU, DEVICE, args.embedding_size, BATCH_SIZE, BACKBONE, calfw, calfw_issame)
            buffer_val(writer, "CALFW", accuracy_calfw, best_threshold_calfw, roc_curve_calfw, epoch + 1)
            accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(MULTI_GPU, DEVICE, args.embedding_size, BATCH_SIZE, BACKBONE, cplfw, cplfw_issame)
            buffer_val(writer, "CPLFW", accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw, epoch + 1)
            accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val(MULTI_GPU, DEVICE, args.embedding_size, BATCH_SIZE, BACKBONE, vgg2_fp, vgg2_fp_issame)
            buffer_val(writer, "VGGFace2_FP", accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp, epoch + 1)
            print("Epoch {}/{}, Evaluation: LFW Acc: {}, CFP_FF Acc: {}, CFP_FP Acc: {}, AgeDB Acc: {}, CALFW Acc: {}, CPLFW Acc: {}, VGG2_FP Acc: {}".format(epoch + 1, NUM_EPOCH, accuracy_lfw, accuracy_cfp_ff, accuracy_cfp_fp, accuracy_agedb, accuracy_calfw, accuracy_cplfw, accuracy_vgg2_fp))
            print("=" * 60)
            sys.stdout.flush() 

            # save checkpoints per epoch
            if MULTI_GPU:
                torch.save(BACKBONE.module.state_dict(), os.path.join(args.model_root, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(args.backbone_name, epoch + 1, batch, get_time())))
            else:
                torch.save(BACKBONE.state_dict(), os.path.join(args.model_root, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(args.backbone_name, epoch + 1, batch, get_time())))
       