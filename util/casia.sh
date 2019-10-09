DATAROOT=/data3/zhengmeisong/TestData/CASIA-IvS-Test/
for pth in `cat ../../py-model/IR50.lst`
do 
    echo $pth
    Model=$pth
    python saveFeatures.py --backbone-resume-root $pth --backbone-name IR_50 \
        --data-root $DATAROOT --list CASIA-IvS-Test-final-v3-revised.lst 
    python test_acc_casia.py  --ftname '.ft' \
        --model $Model --imgRoot $DATAROOT \
        --listFile CASIA-IvS-Test-final-v3-revised.lst --saveFP 0
done
