export OMP_NUM_THREADS=1
DATAROOT=/data_luoqi/zhengmeisong/testData/sw/
MODELROOT=/data_luoqi/zhengmeisong/wkspace/py-model/
MODEL=IR_50
MODEL=IR_18
MODEL=MobileV2
MODEL=IR_SE_152
MODEL=RA_92
for pth in `cat $MODELROOT/$MODEL/$MODEL.lst`
do
    echo $pth
    Model=$MODELROOT/$MODEL/$pth
#    python saveFeatures.py --backbone-resume-root $Model --backbone-name $MODEL \
#       --data-root $DATAROOT --list sw1v1_112.txt --gpu-ids 1
#    python saveFeatures.py --backbone-resume-root $Model --backbone-name $MODEL \
#       --data-root $DATAROOT --list sw1vn_112.txt --gpu-ids 1
    echo $Model
    python testSW1N.py --imgRoot $DATAROOT --model $Model --ftSize 512
done

