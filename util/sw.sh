export OMP_NUM_THREADS=1
DATAROOT=/data03/zhengmeisong/testData/sw/
MODELROOT=/data03/zhengmeisong/wkspace/FR/py-model/
MODEL=IR_18
MODEL=MobileV2
MODEL=IR_SE_152
MODEL=RA_92
MODEL=IR_50
for pth in `cat $MODELROOT/$MODEL/$MODEL.lst`
do
    echo $pth
    Model=$MODELROOT/$MODEL/$pth
    python saveFeatures.py --backbone-resume-root $Model --backbone-name $MODEL \
       --data-root $DATAROOT --list sw1v1_112.txt --gpu-ids 0
    python saveFeatures.py --backbone-resume-root $Model --backbone-name $MODEL \
       --data-root $DATAROOT --list sw1vn_112.txt --gpu-ids 0
    echo $Model
    python testSW1N.py --imgRoot $DATAROOT --model $Model --ftSize 512
done

