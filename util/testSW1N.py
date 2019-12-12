import numpy as np
import os, sys, shutil
import string
import math
import time
import argparse

parser = argparse.ArgumentParser(description='face model evaluate')
parser.add_argument('--ftname', default='.ft', help='name of feat.')
parser.add_argument('--imgRoot', default='/cloud_data01/zhengmeisong/testData/sw/',help = 'imgRoot')
parser.add_argument('--idListFile', default='sw1k_probe.lst', help = 'idFile')
parser.add_argument('--faceListFile', default='sw1w_gallery.lst', help = 'faceFile')
parser.add_argument('--ftRoot', help = 'featRoot')
parser.add_argument('--ftSize', default= 512, type=int, help='model path')
parser.add_argument('--model', default= '', help='model path')

parser.add_argument('--saveFP', type = int, default = 1)

args = parser.parse_args()

def loadFeatureFromModelDir(idListFile, faceListFile,  ftExt = '.arc'):
    ftDir = None
    if not args.model =='None':
        ftDir = args.model.split('.')[0]+'/'
    idList = open(idListFile, 'r').readlines()
    idLabel = np.zeros([len(idList)], dtype = np.int32)
    idFeat = np.zeros([len(idList), args.ftSize],dtype = np.float32)
    for idx,line in enumerate(idList):
        ftName =line.split(' ')[0][:-4]+ftExt
        if not ftDir==None:
            ftElems = ftName.split('/')
            ftName = ftDir + ftElems[-2]+'/'+ftElems[-1]
        if idx%1000==0:
            print('.'),
            sys.stdout.flush()
        try:
            idFeat[idx,:] = np.loadtxt(ftName)[np.newaxis,:]
        except:
            print(ftName, "not detected")
        idLabel[idx] = int( line.split(' ')[-1] )
    
    faceList = open(faceListFile, 'r').readlines()
    faceLabel = np.zeros([len(faceList)], dtype = np.int32)
    faceFeat = np.zeros([len(faceList), args.ftSize],dtype = np.float32)
    for idx,line in enumerate(faceList):
        ftName =line.split(' ')[0][:-4]+ftExt
        if not ftDir==None:
            ftElems = ftName.split('/')
            ftName = ftDir + ftElems[-2]+'/'+ftElems[-1]
        if idx%5000==0:
            print('.'),
            sys.stdout.flush()
        try:
            faceFeat[idx,:] = np.loadtxt(ftName)[np.newaxis,:]
        except:
            print(ftName, "not detected")
        faceLabel[idx] = int( line.split(' ')[-1] )

    return idFeat, faceFeat, idLabel, faceLabel

def saveFP(idListFile, faceListFile, ididx, faceidx, score, th = 0.7):
    if not os.path.isdir('./fp/'):
        os.makedirs('./fp/')
    subdir = './fp/' + 'th_'+str(th) + '/'
    recalled = 'y'
    if score<th:
        recalled = 'n'
    if not os.path.isdir(subdir):
        os.makedirs(subdir)
    with open('./fp/' + 'th_'+str(th) + '.txt', 'a') as fw:
        fw.write(str(ididx)+': ' + str(score)+'\r\n')
        with open(idListFile, 'r') as fp:
            lines = fp.readlines()
            path = lines[ididx].strip().split(' ')[0]
            fw.write(path + '\r\n')
            savename = path.split('/')[-1]
            shutil.copy(path, subdir + recalled + str(ididx) + '_p_' + savename )
        with open(faceListFile, 'r') as fp:
            lines = fp.readlines()
            path = lines[faceidx].strip().split(' ')[0]
            fw.write(path + '\r\n')
            savename = path.split('/')[-1]
            shutil.copy(path, subdir + recalled + str(ididx) + '_g_' + savename )
            ####save real id, should be in ordered in lists####
            path = lines[ididx].strip().split(' ')[0]
            fw.write(path + '\r\n')
            savename = path.split('/')[-1]
            shutil.copy(path, subdir + recalled + str(ididx) + '_r_' + savename )

def saveFN(idListFile, faceListFile, ididx, faceidx, score, th = 0.7):
    if not os.path.isdir('./fn/'):
        os.makedirs('./fn/')
    subdir = './fn/' + 'th_'+str(th) + '/'
    recalled = 'y'
    if score<th:
        recalled = 'n'
    if not os.path.isdir(subdir):
        os.makedirs(subdir)
    with open('./fn/' + 'th_'+str(th) + '.txt', 'a') as fw:
        fw.write(str(ididx)+': ' + str(score)+'\r\n')
        with open(idListFile, 'r') as fp:
            lines = fp.readlines()
            path = lines[ididx].strip().split(' ')[0]
            fw.write(path + '\r\n')
            savename = path.split('/')[-1]
            shutil.copy(path, subdir + recalled + str(ididx) + '_p_' + savename )
        with open(faceListFile, 'r') as fp:
            lines = fp.readlines()
            path = lines[faceidx].strip().split(' ')[0]
            fw.write(path + '\r\n')
            savename = path.split('/')[-1]
            shutil.copy(path, subdir + recalled + str(ididx) + '_g_' + savename )
            

def evaluateAllData(idListFile,faceListFile, base_dir='',ftExt='.arc'):
    idFea, faceFea, idLabel, faceLabel = loadFeatureFromModelDir(idListFile, faceListFile, ftExt=ftExt)
    print(idLabel.shape)
    assert idFea.shape[0] == idLabel.shape[0]
    assert faceFea.shape[0] == faceLabel.shape[0]
    len_face = faceLabel.shape[0]
    fScores = np.zeros(idLabel.shape[0]*faceLabel.shape[0], dtype = np.float32)
    fIsSame = np.zeros(idLabel.shape[0]*faceLabel.shape[0], dtype = np.int32)  
    print("calculate:")
    txtname = args.model.split('.')[0]+ '_sw.txt'
    with open(txtname, 'w') as fw:
        fw.write(args.model)
    
    for t in range(72,79,1):
        th = t/100.0
        acceptedFalse = 0
        selsecedTrue = 0
       
        for idx in range(idLabel.shape[0]):
            scores = (np.tensordot(idFea[idx], faceFea.transpose(), axes=1) + 1)/2
            top1idx = np.argmax(scores)

            if(idLabel[idx]<500) and (scores[top1idx] > th):
                if(faceLabel[top1idx] == idLabel[idx] ):
                    selsecedTrue += 1
                elif args.saveFP:
                    saveFP(idListFile, faceListFile, ididx=idx, faceidx = top1idx, score=scores[top1idx], th=th)
            elif(idLabel[idx]<500) and args.saveFP:
                saveFP(idListFile, faceListFile, ididx=idx, faceidx = top1idx, score = scores[top1idx], th=th)

            if(idLabel[idx]>=500) and (scores[top1idx] > th):
                acceptedFalse +=1
                if args.saveFP:
                    saveFN(idListFile, faceListFile, ididx=idx, faceidx = top1idx, score = scores[top1idx], th=th)
                
                
        print("truelytop1:%3d,acceptedFalse:%3d - tpr:%.2f%%,far:%.2f%% @TH=%.2f"% \
            (selsecedTrue, acceptedFalse, selsecedTrue/5.0, acceptedFalse/5.0, th) )
        with open(txtname, 'a') as fw:
            fw.write("truelytop1:%3d,acceptedFalse:%3d - tpr:%.2f%%,far:%.2f%% @TH=%.2f \n"% \
                     (selsecedTrue, acceptedFalse, selsecedTrue/5.0, acceptedFalse/5.0, th) )


  
def getFalsePositives(fScores, fLabels, thre):
    FPidx= np.where( np.logical_and(fScores >= thre, fLabels == 0) )[0]
    return FPidx

def getFalseNegatives(fScores, fLabels, thre):
    FPidx= np.where( np.logical_and(fScores < thre, fLabels == 1) )[0]
    return FPidx

idListPath = args.imgRoot + args.idListFile
faceListPath = args.imgRoot + args.faceListFile
ftName = args.ftname
evaluateAllData(idListPath,faceListPath, base_dir='',ftExt=ftName)#'.r50_grn3ft_tripletnd')#r50_grn1ft_color')
