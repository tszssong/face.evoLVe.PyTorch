export OMP_NUM_THREADS=4  
DATAROOT=/cloud_data01/StrongRootData/TestData/JA-Test/
MODELROOT=/cloud_data01/zhengmeisong/wkspace/py-model/ol9/
MODELROOT=/cloud_data01/zhengmeisong/wkspace/py-model/olx/
MODEL=RA_92
MODEL=IR_50
MODEL=IR_18
for pth in `cat $MODELROOT/$MODEL/$MODEL.lst`
do 
    echo $pth
    Model=$MODELROOT/$MODEL/$pth
    python saveFeatures.py --backbone-resume-root $Model --backbone-name $MODEL \
        --data-root $DATAROOT --list imgs.lst --gpu-ids 2
    python test_acc_ja.py  --ftname '.ft' \
        --model $Model --imgRoot $DATAROOT \
        --idListFile id.lst  --faceListFile face.lst \
        --saveFP 0
done
