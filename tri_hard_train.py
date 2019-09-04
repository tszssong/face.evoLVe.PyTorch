import os, sys, time, os.path, argparse, socket
import random
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
from util.utils import make_weights_for_balanced_classes, get_val_data,get_val_pair, separate_irse_bn_paras, separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy
from tensorboardX import SummaryWriter
from imgdata.tri_img_iter import TripletImgData
from imgdata.tri_hard_iter import TripletHardImgData
from imgdata.show_img import showBatch
hostname = socket.gethostname()
torch.manual_seed(1337)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='/home/ubuntu/zms/data/ms1m_emore_img')
    parser.add_argument('--model-root', type=str, default='../py-model')
    parser.add_argument('--log-root', type=str, default='../py-log')
    parser.add_argument('--backbone-resume-root', type=str, default='/home/ubuntu/zms/models/ResNet_50_Epoch_33.pth')
    parser.add_argument('--head-resume-root', type=str, default='')
    parser.add_argument('--backbone-name', type=str, default='ResNet_50') # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    parser.add_argument('--input-size', type=str, default="112, 112")
    parser.add_argument('--loss-name', type=str, default='TripletLoss')  # support: ['FocalLoss', 'Softmax', 'TripletLoss']
    parser.add_argument('--embedding-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=60)
    parser.add_argument('--bag-size', type=int, default=1000)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr-stages', type=str, default="0")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num-epoch', type=int, default=1000)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--save-freq', type=int, default=20)
    parser.add_argument('--disp-freq', type=int, default=10)
    parser.add_argument('--test-freq', type=int, default=400)
    args = parser.parse_args()
    writer = SummaryWriter(args.log_root) # writer for buffering intermedium results
    margin = args.margin
    batchSize = args.batch_size
    bagSize = args.bag_size
    
    INPUT_SIZE = [ int(args.input_size.split(',')[0]), int(args.input_size.split(',')[1]) ]
    lrStages = [int(i) for i in args.lr_stages.strip().split(',')]
   
    DEVICE =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    GPU_ID = [int(i) for i in args.gpu_ids.split(",") ]
    print(DEVICE, GPU_ID)
    MULTI_GPU = ( len(GPU_ID)>1 ) # flag to use multiple GPUs
  
    print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    print("=" * 60, "\nOverall Configurations:\n", args)
    BACKBONE = eval(args.backbone_name)(INPUT_SIZE)
    if args.backbone_name.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
     # optionally resume from a checkpoint
    if args.backbone_resume_root and os.path.isfile(args.backbone_resume_root):
        print("Loading Backbone Checkpoint '{}'".format(args.backbone_resume_root))
        # pretrained_net = torch.load(args.backbone_resume_root)
        # for key, v in pretrained_net.items():
        #     print (key)
        BACKBONE.load_state_dict(torch.load(args.backbone_resume_root), strict=False) 
    else:
        print("No Checkpoint, Train from Scratch" )

    if MULTI_GPU:   # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:           # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)

    if args.loss_name == 'Softmax':
        LOSS = nn.CrossEntropyLoss()
    else:
        LOSS = eval(args.loss_name)(args.margin)

    OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn, 'weight_decay': args.weight_decay}, \
                           {'params': backbone_paras_only_bn}], lr = args.lr, momentum = args.momentum)
    # OPTIMIZER = optim.Adam([{'params': backbone_paras_wo_bn, 'weight_decay': args.weight_decay}, \
    #                        {'params': backbone_paras_only_bn}], lr = args.lr)
    print(LOSS,"\n",OPTIMIZER,"\n","="*60, "\n") 
    sys.stdout.flush() 

    # cfp_fp, cfp_fp_issame = get_val_pair(args.data_root, 'cfp_fp')
    jaivs, jaivs_issame = get_val_pair(args.data_root,'ja_ivs.pkl')
    ww1, ww1_issame = get_val_pair(args.data_root,'gl2ms1mdl23f1ww1.pkl')

    train_transform = transforms.Compose([ transforms.Resize([128, 128]),     # smaller side resized
                                           transforms.RandomCrop(INPUT_SIZE),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean =  [0.5, 0.5, 0.5], std =  [0.5, 0.5, 0.5]), ])
   
    dataset_train = TripletHardImgData( os.path.join(args.data_root, 'imgs.lst'), \
                                 input_size = INPUT_SIZE, transform=train_transform)
    train_loader = torch.utils.data.DataLoader( dataset_train, batch_size = args.bag_size, \
                 shuffle=False,  pin_memory = True, num_workers = args.num_workers, drop_last = True )
    print("Number of Training Samples: {}".format(len(train_loader.dataset.samples)))
    sys.stdout.flush()  
    
    print(bagSize, batchSize)
    batch = 0   # batch index
    bagIdx = 0

    for epoch in range(args.num_epoch): # start training process
        for inputs, labels in iter(train_loader):  #bag_data
            bagIdx += 1
            for l_idx in range(len(lrStages)):
                if bagIdx == lrStages[l_idx]:
                    schedule_lr(OPTIMIZER)
            start = time.time()
            features = torch.empty(bagSize, args.embedding_size)
            BACKBONE.eval()  # set to testing mode

            for b_idx in range(int(bagSize/batchSize)):
                batchIn = inputs[b_idx*batchSize:(b_idx+1)*batchSize,:]
                batchFea = BACKBONE(batchIn.to(DEVICE))
                batchFea = F.normalize(batchFea).detach()
                features[b_idx*batchSize:(b_idx+1)*batchSize,:] = batchFea
           
            dist_matrix = torch.mm(features, features.t())
            dist_matrix = 2-dist_matrix
            dist_matrix = dist_matrix.numpy()
            assert dist_matrix.shape[0] == bagSize

            np.set_printoptions(suppress=True)
            baglabel_1v = labels.view(labels.shape[0]).numpy().astype(np.int64)  #longTensor=int64
           
            nCount = 0    #number of valid triplets
            bagList = []
            aCount = 0
            last_a_label = 0
            for a_idx in range( bagSize ):
                a_label = baglabel_1v[a_idx]
                if last_a_label==a_label:
                    aCount +=1
                else:
                    aCount = 0
                if aCount > 7:
                    continue
                last_a_label = a_label
                p_candidate = np.where(baglabel_1v==a_label)[0]
                p_candidate = p_candidate[p_candidate>a_idx]
                np.random.shuffle(p_candidate)
                pCount = 0
                for p_idx in p_candidate:
                    pCount += 1       # positive index 7*7 TODO
                    if pCount > 7:
                        break
                    pDist = dist_matrix[a_idx][p_idx]
                    distTh = pDist + args.margin
                    n_candidate =  np.where( np.logical_and( dist_matrix[a_idx]<distTh, 
                                                             dist_matrix[a_idx]>pDist, 
                                                             baglabel_1v!=a_label) )[0]
                    
                    if(n_candidate.size<=0):
                        continue
                    else:
                        n_idx = np.random.choice(n_candidate)
                    bagList.append((a_idx, p_idx, n_idx))
                    nCount+=1
                    
            BACKBONE.train()  # set to training mode
            losses = AverageMeter()
            acc   = AverageMeter()
            for b_idx in range(int(nCount/batchSize)): 
                batch_idx = bagList[b_idx*batchSize:(b_idx+1)*batchSize]
                batch_idx = np.array(batch_idx).flatten()
                # print(batch_idx)
                bIn = inputs[batch_idx].to(DEVICE)
                bLabel = labels[batch_idx].to(DEVICE)
                                
                outputs = BACKBONE(bIn)
                # show batch data: only ubuntu
                if hostname=="ubuntu-System-Product-Name":
                    features = F.normalize(outputs).detach()
                    showBatch(bIn.cpu().numpy(), bLabel.cpu().numpy(), \
                              features.cpu().numpy(), show_x=args.batch_size)
           
                loss, loss_batch = LOSS(outputs, bLabel, DEVICE, margin)
                loss_batch = loss_batch.detach().cpu().numpy()
                n_err = np.where(loss_batch>0)[0].shape[0] 
                prec = 1.0 - float(n_err) / loss_batch.shape[0]
                
                losses.update(loss.data.item(), bIn.size(0))
                acc.update(prec, bIn.size(0))
                # compute gradient and do SGD step
                OPTIMIZER.zero_grad()
                loss.backward()
                OPTIMIZER.step()
                batch += 1 # batch index
                if batch%args.disp_freq==0:
                    print('Batch: {}\t' 'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                        'Prec {acc.val:.3f} ({acc.avg:.3f})'.format(batch, loss=losses, acc=acc))
                    
            bag_loss = losses.avg
            bag_acc = acc.avg
            writer.add_scalar("Training_Loss", bag_loss, epoch + 1)
            writer.add_scalar("Training_Accuracy", bag_acc, epoch + 1)
            print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), \
                  " Bag:%d - %d, Batch:%d "%(bagIdx, nCount, batch), "%.3f s/bag"%(time.time()-start))
       
            print('Epoch: {}/{} \t' 'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                  'Prec {acc.val:.3f} ({acc.avg:.3f})'.format(epoch+1, args.num_epoch, loss=losses, acc=acc))
            print("=" * 60)
            sys.stdout.flush() 

            if (bagIdx%args.test_freq==0 and bagIdx!=0):
                print("\nEvaluation on JA_IVS......")
                sys.stdout.flush()
                accuracy_jaivs, best_threshold_jaivs = perform_val(MULTI_GPU, DEVICE,     \
                                    args.embedding_size, args.batch_size, BACKBONE, jaivs, jaivs_issame)
                buffer_val(writer, "JA_IVS", accuracy_jaivs, best_threshold_jaivs, epoch + 1)
                print("Epoch %d/%d, JA_IVS: %.4f"%(epoch + 1, args.num_epoch,accuracy_jaivs) )

                # print("=" * 60, "\nEvaluation on CFP_FP......")
                # sys.stdout.flush()
                # accuracy_cfp_fp, best_threshold_cfp_fp = perform_val(MULTI_GPU, DEVICE,  \
                #                     args.embedding_size, args.batch_size, BACKBONE, cfp_fp, cfp_fp_issame)
                # buffer_val(writer, "CFP_FP", accuracy_cfp_fp, best_threshold_cfp_fp, epoch + 1)
                # print("Epoch %d/%d, CFP_FP: %.4f,"%(epoch + 1, args.num_epoch, accuracy_cfp_fp) )

                print("=" * 60, "\nEvaluation on gl2ms1mdl23f1ww1......")
                accuracy_ww1, best_threshold_ww1 = perform_val(MULTI_GPU, DEVICE,        \
                                    args.embedding_size, args.batch_size, BACKBONE, ww1, ww1_issame)
                buffer_val(writer, "WW1", accuracy_ww1, best_threshold_ww1, epoch + 1)
                print("Epoch %d/%d, gmcf Acc: %.4f"%(epoch + 1, args.num_epoch, accuracy_ww1) )
                sys.stdout.flush() 

            if (bagIdx%args.save_freq==0 and bagIdx!=0):
                print("Save Checkpoints Batch %d..."%batch)
                if MULTI_GPU:
                    torch.save(BACKBONE.module.state_dict(), os.path.join(args.model_root, \
                              "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth"       \
                              .format(args.backbone_name, epoch + 1, batch, get_time())))
                else:
                    torch.save(BACKBONE.state_dict(), os.path.join(args.model_root,        \
                              "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth"       \
                              .format(args.backbone_name, epoch + 1, batch, get_time())))
       
