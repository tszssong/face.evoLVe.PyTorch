DATAROOT=/cloud_data01/zhengmeisong/TestData/JA-Test/
MODELROOT=/cloud_data01/zhengmeisong/wkspace/olx/py-model/
MODEL=IR_18
for pth in `cat $MODELROOT/$MODEL.lst`
do 
    echo $pth
    Model=$MODELROOT/$pth
    python saveFeatures.py --backbone-resume-root $Model --backbone-name $MODEL \
        --data-root $DATAROOT --list imgs.lst --gpu-ids 2
    python test_acc_ja.py  --ftname '.ft' \
        --model $Model --imgRoot $DATAROOT \
        --idListFile id.lst  --faceListFile face.lst \
        --saveFP 0
done
