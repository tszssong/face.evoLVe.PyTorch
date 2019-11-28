import numpy as np
import os, sys, shutil
import string
import math
import time
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='face model evaluate')
parser.add_argument('--ftname', default='.ft', help='name of feat.')
parser.add_argument('--imgRoot', help = 'imgRoot')
parser.add_argument('--idListFile', help = 'idFile')
parser.add_argument('--faceListFile', help = 'faceFile')
parser.add_argument('--ftRoot', help = 'featRoot')
parser.add_argument('--model', default= 'None', help='model path')
parser.add_argument('--saveFP', type = int, default = 1)
FARs=[1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]

args = parser.parse_args()

def loadFeatureFromModelDir(idListFile, faceListFile,  ftExt = '.arc'):
    ftDir = None
    if not args.model =='None':
        ftDir = args.model.split('.')[0]+'/'

    idList = open(idListFile, 'r').readlines()
    idLabel = np.zeros([len(idList)], dtype = np.int32)
    idFeat = np.zeros([len(idList), 512],dtype = np.float32)
#    for idx,line in enumerate(idList):
    for idx,line in tqdm(enumerate(idList)):
        ftName =line.split(' ')[0][:-4]+ftExt
        if not ftDir==None:
            ftElems = ftName.split('/')
            ftName = ftDir + ftElems[-2]+'/'+ftElems[-1]
        idFeat[idx,:] = np.loadtxt(ftName)[np.newaxis,:]
        idLabel[idx] = int( line.split(' ')[-1] )
    
    faceList = open(faceListFile, 'r').readlines()
    faceLabel = np.zeros([len(faceList)], dtype = np.int32)
    faceFeat = np.zeros([len(faceList), 512],dtype = np.float32)
    #for idx,line in enumerate(faceList):
    for idx,line in tqdm(enumerate(faceList)):
        ftName =line.split(' ')[0][:-4]+ftExt
        if not ftDir==None:
            ftElems = ftName.split('/')
            ftName = ftDir + ftElems[-2]+'/'+ftElems[-1]
        #if idx%1000==0:
        #    print('.',end=' ')
        #    sys.stdout.flush()
        faceFeat[idx,:] = np.loadtxt(ftName)[np.newaxis,:]
        faceLabel[idx] = int( line.split(' ')[-1] )

    return idFeat, faceFeat, idLabel, faceLabel

def evaluateAllData(idListFile,faceListFile, base_dir='',ftExt='.arc'):
    idFea, faceFea, idLabel, faceLabel = loadFeatureFromModelDir(idListFile, faceListFile, ftExt=ftExt)
    print(idLabel.shape)
    assert idFea.shape[0] == idLabel.shape[0]
    assert faceFea.shape[0] == faceLabel.shape[0]
    len_face = faceLabel.shape[0]
    fScores = np.zeros(idLabel.shape[0]*faceLabel.shape[0], dtype = np.float32)
    fIsSame = np.zeros(idLabel.shape[0]*faceLabel.shape[0], dtype = np.int32)  
    print("calculate:")
    #for idx in range(idLabel.shape[0]):
    for idx in tqdm(range(idLabel.shape[0])):
        scores = (np.tensordot(idFea[idx], faceFea.transpose(), axes=1) + 1)/2
        isSame = np.zeros( faceLabel.shape[0], dtype = np.int32)
        isSame[np.where(faceLabel==idLabel[idx])] = 1
        fScores[idx*len_face:(idx+1)*len_face] = scores
        fIsSame[idx*len_face:(idx+1)*len_face] = isSame
    #    if(idx%1000==0):
    #        print(".",end=' ')
    minScore = np.min(fScores)
    maxScore = np.max(fScores)
    print("maxScore, minSocre:",maxScore, minScore)
    stepThr = max((maxScore - minScore) / 1000, 0.0001)
    GenuNum = np.where(fIsSame == 1)[0].shape[0]
    ImpoNum = np.where(fIsSame == 0)[0].shape[0]
    return fScores,fIsSame,minScore,maxScore,stepThr,GenuNum,ImpoNum

def Find_Kth_max(array,k):
    for i in range(1,k):
        for j in range(i,0,-1):
            if array[j] > array[j-1]:
                array[j],array[j-1] = array[j-1],array[j]
            else:
                pass
    for i in range(k,len(array)):
        if array[i] > array[k-1]:
            array[k-1] = array[i]
            for j in range(k-1,0,-1):
                if array[j] > array[j-1]:
                    array[j],array[j-1] = array[j-1],array[j]
                else:
                    pass
    return array[k-1]
            
def getRocCurveV2(fScores,fLabels,minScore,maxScore,stepThr,GenuNum,ImpoNum):
    ther = minScore
    TPRList = []
    FARList = []
    AccList = []
    PairNum=fScores.shape[0]
    ThrList = []
    
    negLabels = np.where(fLabels==0)[0]
    print(negLabels.shape)
    nNegs = negLabels.shape[0]
    nfScores = fScores[negLabels]
    fsort = np.argsort(-nfScores)
    #fsort = np.sort(-nfScores)

    for idx,far in enumerate(FARs):
        negNums = int(round(ImpoNum*far))
        if negNums == 0:
            negNums = 1

        fsortx=fsort[:negNums]
        ther = nfScores[fsortx[-1]]

        CurrTPNum=len(np.where(fScores*fLabels>=ther)[0])
        CurrFPNum = len(np.where(fScores * (fLabels-1)*(-1) >= ther)[0])

        CurrTPR = float(CurrTPNum)/GenuNum
        CurrFAR = float(CurrFPNum)/ImpoNum

        CurrAcc = np.float( CurrTPNum + ( ImpoNum-CurrFPNum ) ) / PairNum
        TPRList.append(CurrTPR)
        FARList.append(CurrFAR)
        AccList.append(CurrAcc)
        ThrList.append(ther.copy())

    FARArry = np.asarray(FARList)
    TPRArry = np.asarray(TPRList)
    AccArry = np.asarray(AccList)
    ThrArry = np.asarray(ThrList)

    return FARArry, TPRArry,AccArry,ThrArry

def getFalsePositives(fScores, fLabels, thre):
    FPidx= np.where( np.logical_and(fScores >= thre, fLabels == 0) )[0]
    return FPidx

def getFalseNegatives(fScores, fLabels, thre):
    FPidx= np.where( np.logical_and(fScores < thre, fLabels == 1) )[0]
    return FPidx

def getFarValues(FARs,FARArry,TPRArry,AccArry,ThrArry):
    FARArrys=np.repeat(FARArry[np.newaxis,:],len(FARs),axis=0)
    distances = np.abs(FARArrys-np.asarray(FARs)[:,np.newaxis])
    minIdxs = np.argmin(distances,axis=1)
    rFARs = FARArry[minIdxs]
    TPRs = TPRArry[minIdxs]
    Thrs = ThrArry[minIdxs]
    ACCs = AccArry[minIdxs]
    print(args.model)
    if 'Rev' in args.imgRoot:
        txtname = args.model.split('.')[0]+ '_ja_rev.txt'
    else:
        txtname = args.model.split('.')[0]+ '_ja.txt'
    with open(txtname, 'w') as fw:
        fw.write(args.model + '\n')
        for idx,far in enumerate(FARs):
            print('%.9f: %f @ %f'%(far,TPRs[idx],Thrs[idx]))
            fw.write('%.9f: %f @ %f\n'%(far,TPRs[idx],Thrs[idx]))
    return Thrs

idListPath = args.imgRoot + args.idListFile
faceListPath = args.imgRoot + args.faceListFile
ftName = args.ftname #'.dl23f1-80' #'.q2-146'
fScores,fLabels,minScore,maxScore,stepThr,GenuNum,ImpoNum=evaluateAllData(idListPath,faceListPath, base_dir='',ftExt=ftName)#'.r50_grn3ft_tripletnd')#r50_grn1ft_color')
minScore = np.max([minScore, 0.5])
maxScore = np.min([maxScore,0.9])
FARArry, TPRArry,AccArry,ThrArry=getRocCurveV2(fScores,fLabels,minScore,maxScore,0.0025,GenuNum,ImpoNum)
THRS=getFarValues(FARs,FARArry,TPRArry,AccArry,ThrArry)

idList = open(idListPath, 'r').readlines()
faceList = open(faceListPath, 'r').readlines()
if args.saveFP==1:
    if os.path.exists("./log/"):
        shutil.rmtree("./log/")
        os.mkdir("./log/")
    for idx, thr in enumerate(THRS):
        if idx ==0 or idx == 1:
            continue
        fp_pair_idxs  = getFalsePositives(fScores, fLabels,thr)
        fn_pair_idxs = getFalseNegatives(fScores, fLabels,thr)
        # print(fp_pair_idxs, fn_pair_idxs)
        print(thr,'fpairs',len(fp_pair_idxs),'npairs', len(fn_pair_idxs))
        fpName = './log/fpIdx_%d.txt'%idx
        fnName = './log/fnIdx_%d.txt'%idx
        fpDir  = './log/fpIdx_%d'%idx +'/'
        fnDir  = './log/fnIdx_%d'%idx +'/'
        if not os.path.exists(fpDir):
            os.mkdir(fpDir)
        if not os.path.exists(fnDir):
            os.mkdir(fnDir)
        fpw = open(fpName,'w')
        fnw = open(fnName,'w')
        for pair_idx in list(fp_pair_idxs):
            # print("fp:",pair_idx)
            score = fScores[pair_idx]
            id_idx = pair_idx / len(faceList)
            face_idx = pair_idx % len(faceList)
            # print(idList[id_idx])
            # print(faceList[face_idx])
            idPath = idList[id_idx].split(' ')[0]
            facePath = faceList[face_idx].split(' ')[0]
            fpw.write("%d\t%6.5f\t%s\t%s\n"%(pair_idx,score,idPath,facePath))
            shutil.copy(idPath, fpDir + str(pair_idx) + '_id.jpg')
            shutil.copy(facePath, fpDir + str(pair_idx) + '_face.jpg')
        for pair_idx in list(fn_pair_idxs):
            # print("fn:",pair_idx)
            score = fScores[pair_idx]
            id_idx = pair_idx / len(faceList)
            face_idx = pair_idx % len(faceList)
            # print(idList[id_idx])
            # print(faceList[face_idx])
            idPath = idList[id_idx].split(' ')[0]
            facePath = faceList[face_idx].split(' ')[0]
            fnw.write("%d\t%6.5f\t%s\t%s\n"%(pair_idx,score,idPath,facePath))
            shutil.copy(idPath, fnDir + str(pair_idx) + '_id.jpg')
            shutil.copy(facePath, fnDir + str(pair_idx) + '_face.jpg')
           
        fpw.close()
        fnw.close()
