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
    parser.add_argument('--backbone-resume-root', type=str, default='../home/ubuntu/zms/models/ResNet_50_Epoch_33.pth')
    parser.add_argument('--head-resume-root', type=str, default='')
    parser.add_argument('--backbone-name', type=str, default='ResNet_50') # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    parser.add_argument('--input-size', type=str, default="112, 112")
    parser.add_argument('--loss-name', type=str, default='TripletLoss')  # support: ['FocalLoss', 'Softmax', 'TripletLoss']
    parser.add_argument('--embedding-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=120)
    parser.add_argument('--bag-size', type=int, default=600)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr-stages', type=str, default="0")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num-epoch', type=int, default=1000)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--save-freq', type=int, default=20)
    parser.add_argument('--test-freq', type=int, default=400)
    args = parser.parse_args()
    writer = SummaryWriter(args.log_root) # writer for buffering intermedium results
    margin = args.margin
    batchSize = args.batch_size
    bagSize = args.bag_size
    assert bagSize%batchSize == 0
    # assert batchSize%3 == 0 #triplet
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
        BACKBONE.load_state_dict(torch.load(args.backbone_resume_root)) 
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

    # OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn, 'weight_decay': args.weight_decay}, \
    #                        {'params': backbone_paras_only_bn}], lr = args.lr, momentum = args.momentum)
    OPTIMIZER = optim.Adam([{'params': backbone_paras_wo_bn, 'weight_decay': args.weight_decay}, \
                           {'params': backbone_paras_only_bn}], lr = args.lr)
    print(LOSS,"\n",OPTIMIZER,"\n","="*60, "\n") 
    sys.stdout.flush() 

    cfp_fp, cfp_fp_issame = get_val_pair(args.data_root, 'cfp_fp')
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
           
            bagIn = torch.empty(bagSize*3,3,INPUT_SIZE[0],INPUT_SIZE[1])
            bagLabel = torch.LongTensor(bagSize*3, 1)
            for a_idx in range( bagSize ):
                p_dist = dist_matrix[a_idx].copy()
                n_dist = dist_matrix[a_idx]
                a_label = baglabel_1v[a_idx]
                
                # TODO:  skip 1 img/id
                if(np.sum(baglabel_1v==a_label) == 1):
                    p_idx = a_idx
                # elif(np.sum(baglabel_1v==a_label) == 2):
                #     p_dist[np.where(baglabel_1v!=a_label)] = 0
                #     p_idx = p_dist.argmax()
                else:
                    p_dist[np.where(baglabel_1v!=a_label)] = 0
                    # p_dist[p_dist.argmax()]=0
                    p_idx = p_dist.argmax()
                
                # TODO: incase batch_size < class id images
                n_dist[ np.where(baglabel_1v==a_label) ] = 8192 #np.NaN    #fill same ids with a bigNumber
                # r=random.randint(0,3)
                # for i in range(r):
                #     n_dist[n_dist.argmin()]=8192
                n_idx = n_dist.argmin()
                
                bagIn[a_idx*3]   = inputs[a_idx]
                bagIn[a_idx*3+1] = inputs[p_idx]
                bagIn[a_idx*3+2] = inputs[n_idx]
                bagLabel[a_idx*3]   = labels[a_idx]
                bagLabel[a_idx*3+1] = labels[p_idx]
                bagLabel[a_idx*3+2] = labels[n_idx]
                
            BACKBONE.train()  # set to training mode
            losses = AverageMeter()
            acc   = AverageMeter()
            for b_idx in range(int(bagSize/batchSize)): 
                _begin = int(3*b_idx*batchSize)
                _end = int(3*(b_idx+1)*batchSize)
                bIn = bagIn[_begin:_end,:].to(DEVICE)
                bLabel = bagLabel[_begin:_end, :].to(DEVICE)
                
                outputs = BACKBONE(bIn)
                # show batch data: only ubuntu
                if hostname=="ubuntu-System-Product-Name":
                    features = F.normalize(outputs).detach()
                    showBatch(bIn.cpu().numpy(), bLabel.cpu().numpy(), \
                              features.cpu().numpy(), show_x=args.batch_size)
           
                loss, loss_batch = LOSS(outputs, bLabel, DEVICE, margin)
                loss_batch = loss_batch.detach().cpu().numpy()
                n_err = np.where(loss_batch!=0)[0].shape[0] 
                prec = 1.0 - float(n_err) / loss_batch.shape[0]
                
                losses.update(loss.data.item(), bIn.size(0))
                acc.update(prec, bIn.size(0))
                # compute gradient and do SGD step
                OPTIMIZER.zero_grad()
                loss.backward()
                OPTIMIZER.step()
                batch += 1 # batch index
          
            bag_loss = losses.avg
            bag_acc = acc.avg
            writer.add_scalar("Training_Loss", bag_loss, epoch + 1)
            writer.add_scalar("Training_Accuracy", bag_acc, epoch + 1)
            print( time.strftime("%Y-%m-%d %H:%M:%S\t", time.localtime()), \
                  " Bag:%d Batch:%d\t"%(bagIdx, batch), "%.3f s/bag"%(time.time()-start))
            # print("loss=%.4f, acc=%.4f"%(loss, prec))
            print('Epoch: {}/{} \t' 'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                  'Prec {acc.val:.3f} ({acc.avg:.3f})'.format(epoch+1, args.num_epoch, loss=losses, acc=acc))
            print("=" * 60)
            sys.stdout.flush() 

            if (bagIdx%args.test_freq==0 and bagIdx!=0):
                print("=" * 60, "\nEvaluation on CFP_FP, JA_IVS, gl2ms1mdl23f1ww1......")
                sys.stdout.flush()
                accuracy_jaivs, best_threshold_jaivs = perform_val(MULTI_GPU, DEVICE,     \
                                    args.embedding_size, args.batch_size, BACKBONE, jaivs, jaivs_issame)
                buffer_val(writer, "JA_IVS", accuracy_jaivs, best_threshold_jaivs, epoch + 1)

                accuracy_cfp_fp, best_threshold_cfp_fp = perform_val(MULTI_GPU, DEVICE,  \
                                    args.embedding_size, args.batch_size, BACKBONE, cfp_fp, cfp_fp_issame)
                buffer_val(writer, "CFP_FP", accuracy_cfp_fp, best_threshold_cfp_fp, epoch + 1)

                accuracy_ww1, best_threshold_ww1 = perform_val(MULTI_GPU, DEVICE,        \
                                    args.embedding_size, args.batch_size, BACKBONE, ww1, ww1_issame)
                buffer_val(writer, "WW1", accuracy_ww1, best_threshold_ww1, epoch + 1)

                print("Epoch %d/%d, CFP_FP: %.4f, JA_IVS: %.4f, WW1 Acc: %.4f" \
                    %(epoch + 1, args.num_epoch, accuracy_cfp_fp, accuracy_jaivs, accuracy_ww1))
                print("=" * 60)
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
       
