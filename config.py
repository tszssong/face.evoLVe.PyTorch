import torch

configurations = {
    1: dict(
        SEED = 1337, # random seed for reproduce results

        # DATA_ROOT = '/cloud_data01/zhengmeisong/data/ms1m_emore_img/', # the parent root where your train/val/test data are stored
        DATA_ROOT = '/home/zhengmeisong/data/ms1m-retinaface-t1-img/',
        MODEL_ROOT = '../py-model', # the root to buffer your checkpoints
        LOG_ROOT = '../py-log', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = '../py-model/backbone_ir50_ms1m_epoch120.pth', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = './', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'RA_92', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        # BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'Softmax',       # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Softmax',   # support: ['Focal', 'Softmax', 'Triplet']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 480,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 15, # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        # STAGES = [5,10,15,25], # epoch stages to decay learning rate
        STAGES = [3, 6, 9, 12], # epoch stages to decay learning rate

        MULTI_GPU = True, # flag to use multiple GPUs; 
        # DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu"),
        # GPU_ID = [4,5,6,7], # specify your GPU ids
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        GPU_ID = [0,1,2,3,4,5,6,7], # specify your GPU ids
        PIN_MEMORY = True,
        NUM_WORKERS = 0,
    ),
}

