DATAROOT=/data3/zhengmeisong/TestData/JA-Test/
for pth in `cat ../../py-model/IR50.lst`
do 
    echo $pth
    Model=$pth
    python saveFeatures.py --backbone-resume-root $pth --backbone-name IR_50 \
        --data-root $DATAROOT --list imgs.lst --gpu-ids 2
    python test_acc_ja.py  --ftname '.ft' \
        --model $Model --imgRoot $DATAROOT \
        --idListFile id.lst  --faceListFile face.lst \
        --saveFP 0
done
