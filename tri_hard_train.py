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
import torch.nn.functional as F
# torch.multiprocessing.set_sharing_strategy('file_system')
hostname = socket.gethostname()
torch.manual_seed(1337)

#https://blog.csdn.net/Tan_HandSome/article/details/82501902
def get_dist( emb ):
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
    parser.add_argument('--batch-size', type=int, default=120)
    parser.add_argument('--bag-size', type=int, default=600)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr-stages', type=str, default="120000, 165000, 195000")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num-epoch', type=int, default=1000)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--disp-freq', type=int, default=1)
    parser.add_argument('--test-bag', type=int, default=20)
    args = parser.parse_args()
    writer = SummaryWriter(args.log_root) # writer for buffering intermedium results
    margin = args.margin
    batchSize = args.batch_size
    bagSize = args.bag_size
    assert bagSize%batchSize == 0
    assert batchSize%3 == 0 #triplet
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
   
    dataset_train = TripletHardImgData( os.path.join(args.data_root, 'imgs.lst'), \
                                 input_size = INPUT_SIZE, transform=train_transform)
        
    # dataset_train = TripletHardImgData(os.path.join(args.data_root, 'imgs'), model, transform=train_transform, use_list=False)
    train_loader = torch.utils.data.DataLoader( dataset_train, batch_size = args.bag_size, \
                 shuffle=False,  pin_memory = True, num_workers = args.num_workers, drop_last = True )
    print("Number of Training Samples: {}".format(len(train_loader.dataset.samples)))
    sys.stdout.flush()  
    
    print(bagSize, batchSize)
    batch = 0   # batch index
    bagIdx = 0
    for epoch in range(args.num_epoch): # start training process
        for l_idx in range(len(lrStages)):
            if epoch == lrStages[l_idx]:
                schedule_lr(OPTIMIZER)
        losses = AverageMeter()
        acc   = AverageMeter()
        for inputs, labels in iter(train_loader):  #bag_data
            start = time.time()
            bagIdx += 1
            features = torch.empty(bagSize, args.embedding_size)
            BACKBONE.eval()  # set to training mode

            for b_idx in range(int(bagSize/batchSize)):
                batchIn = inputs[b_idx*batchSize:(b_idx+1)*batchSize,:]
                batchFea = BACKBONE(batchIn.to(DEVICE))
                batchFea = F.normalize(batchFea).detach()
                features[b_idx*batchSize:(b_idx+1)*batchSize,:] = batchFea
           
            dist_matrix = torch.empty(bagSize, bagSize)
            dist_matrix = get_dist(features.cpu().numpy())
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
                else:
                    p_dist[np.where(baglabel_1v!=a_label)] = 0
                    numCandidate = int( max(1, 0.5*np.where(baglabel_1v==a_label)[0].shape[0]) )
                    p_candidate = p_dist.argsort()[-numCandidate:]
                    p_idx = np.random.choice( p_candidate )
                
                # TODO: incase batch_size < class id images
                n_dist[ np.where(baglabel_1v==a_label) ] = 2048    #fill same ids with a bigNumber
                numCandidate = int( max(1, bagSize*0.1) )
                # numCandidate = 1
                n_candidate = n_dist.argsort()[ :numCandidate ]    
                n_idx = np.random.choice(n_candidate)
                bagIn[a_idx*3]   = inputs[a_idx]
                bagIn[a_idx*3+1] = inputs[p_idx]
                bagIn[a_idx*3+2] = inputs[n_idx]
                bagLabel[a_idx*3]   = labels[a_idx]
                bagLabel[a_idx*3+1] = labels[p_idx]
                bagLabel[a_idx*3+2] = labels[n_idx]
                
            BACKBONE.train()  # set to training mode
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
                
                losses.update(loss.data.item(), inputs.size(0))
                acc.update(prec, inputs.size(0))
                # compute gradient and do SGD step
                OPTIMIZER.zero_grad()
                loss.backward()
                OPTIMIZER.step()
                batch += 1 # batch index
          
            bag_loss = losses.avg
            bag_acc = acc.avg
            writer.add_scalar("Training_Loss", bag_loss, epoch + 1)
            writer.add_scalar("Training_Accuracy", bag_acc, epoch + 1)
            print( "epoch:",epoch, time.strftime("%Y-%m-%d %H:%M:%S\t", time.localtime()), "%.3f s/bag"%(time.time()-start) )
            print('Epoch: {}/{} Bag: {} Batch: {} \t'
                'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                'Prec {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch + 1, args.num_epoch,bagIdx, batch, loss=losses, acc=acc))
            print("=" * 60)
            sys.stdout.flush() 

            if (bagIdx%args.test_bag==0 and bagIdx!=0):
            #     # perform validation & save checkpoints per epoch
                print("=" * 60, "\nPerform Evaluation on CFP_FP AgeDB, and Save Checkpoints...")
                sys.stdout.flush() 
                buffer_val(writer, "CFP_FP", accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp, epoch + 1)
                accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(MULTI_GPU, DEVICE, args.embedding_size, args.batch_size, BACKBONE, agedb, agedb_issame)
                buffer_val(writer, "AgeDB", accuracy_agedb, best_threshold_agedb, roc_curve_agedb, epoch + 1)
                accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(MULTI_GPU, DEVICE, args.embedding_size, args.batch_size, BACKBONE, calfw, calfw_issame)
                print("Epoch {}/{}, Evaluation: CFP_FP Acc: {}, AgeDB Acc: {}".format(epoch + 1, \
                       args.num_epoch, accuracy_cfp_fp, accuracy_agedb))
                print("=" * 60)
                sys.stdout.flush() 

                # print("=" * 60, "\nPerform Evaluation on LFW, CFP_FF, CFP_FP, AgeDB, CALFW, CPLFW and VGG2_FP, and Save Checkpoints...")
                # sys.stdout.flush() 
                # accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, args.embedding_size, args.batch_size, BACKBONE, lfw, lfw_issame)
                # buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, epoch + 1)
                # accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(MULTI_GPU, DEVICE, args.embedding_size, args.batch_size, BACKBONE, cfp_ff, cfp_ff_issame)
                # buffer_val(writer, "CFP_FF", accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff, epoch + 1)
                # accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(MULTI_GPU, DEVICE, args.embedding_size, args.batch_size, BACKBONE, cfp_fp, cfp_fp_issame)
                # buffer_val(writer, "CFP_FP", accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp, epoch + 1)
                # accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(MULTI_GPU, DEVICE, args.embedding_size, args.batch_size, BACKBONE, agedb, agedb_issame)
                # buffer_val(writer, "AgeDB", accuracy_agedb, best_threshold_agedb, roc_curve_agedb, epoch + 1)
                # accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(MULTI_GPU, DEVICE, args.embedding_size, args.batch_size, BACKBONE, calfw, calfw_issame)
                # buffer_val(writer, "CALFW", accuracy_calfw, best_threshold_calfw, roc_curve_calfw, epoch + 1)
                # accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(MULTI_GPU, DEVICE, args.embedding_size, args.batch_size, BACKBONE, cplfw, cplfw_issame)
                # buffer_val(writer, "CPLFW", accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw, epoch + 1)
                # accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val(MULTI_GPU, DEVICE, args.embedding_size, args.batch_size, BACKBONE, vgg2_fp, vgg2_fp_issame)
                # buffer_val(writer, "VGGFace2_FP", accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp, epoch + 1)
                # print("Epoch {}/{}, Evaluation: LFW Acc: {}, CFP_FF Acc: {}, CFP_FP Acc: {}, AgeDB Acc: {}, CALFW Acc: {}, CPLFW Acc: {}, VGG2_FP Acc: {}".format(epoch + 1, args.num_epoch, accuracy_lfw, accuracy_cfp_ff, accuracy_cfp_fp, accuracy_agedb, accuracy_calfw, accuracy_cplfw, accuracy_vgg2_fp))
                # print("=" * 60)
                # sys.stdout.flush() 

                if MULTI_GPU:
                    torch.save(BACKBONE.module.state_dict(), os.path.join(args.model_root, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(args.backbone_name, epoch + 1, batch, get_time())))
                else:
                    torch.save(BACKBONE.state_dict(), os.path.join(args.model_root, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(args.backbone_name, epoch + 1, batch, get_time())))
       