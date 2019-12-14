export OMP_NUM_THREADS=4  
DATAROOT=/data3/zhengmeisong/TestData/JA-Rev-Test/
MODELROOT=/data3/zhengmeisong/wkspace/py-model/
MODEL=IR_50
MODEL=IR_18
MODEL=MobileV2
MODEL=RA_92
for pth in `cat $MODELROOT/$MODEL/$MODEL.lst`
do 
    echo $pth
    Model=$MODELROOT/$MODEL/$pth
    python saveFeatures.py --backbone-resume-root $Model --backbone-name $MODEL \
        --data-root $DATAROOT --list imgs.lst --gpu-ids 1
    python test_acc_ja.py  --ftname '.ft' \
        --model $Model --imgRoot $DATAROOT \
        --idListFile id.lst  --faceListFile face.lst \
        --saveFP 0
done
