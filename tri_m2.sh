  #--teacher-resume-root ../py-preTrain/Backbone_IR_50_Epoch_3_Batch_228058_Time_2019-12-03-05-01_checkpoint.pth \
export OMP_NUM_THREADS=4
NET=MobileV2
mkdir ../py-model/${NET}
python triKD.py --data-root /data03/zhengmeisong/data/jr-pairs/ \
  --teacher-name IR_50 \
  --teacher-resume-root ../py-preTrain/Backbone_IR_50_Epoch_3_Batch_355997_Time_2019-12-08-12-31_checkpoint.pth \
  --model-root ../py-model/${NET}/ \
  --num-workers 0 --num-epoch 100 \
  --bag-size 72000 --batch-size 120  --test-freq 50 --save-freq 5000\
  --lr 0.01 --lr-stages 1800,4800 --margin 0.5 --alpha 1.0\
  --backbone-name ${NET} \
  --backbone-resume-root ../py-preTrain/Backbone_MobileV2_Epoch_20_Batch_668200_Time_2019-11-30-15-49_checkpoint.pth \
  --gpu-ids 1,2,3 \
  2>&1 | tee ../py-log/tri_`date +'%m_%d-%H_%M'`.log  

