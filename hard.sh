  #--backbone-resume-root ../py-model/Backbone_ResNet_50_Epoch_1_Batch_9801_Time_2019-09-27-13-03_checkpoint.pth \
python tri_hard_train.py --data-root /data3/zhengmeisong/data/gl2ms1m_dl23W1f1_150WW1_img/ \
  --num-workers 4 --num-epoch 100 \
  --bag-size 48000 --batch-size 1200 --test-freq 50 --save-freq 50\
  --lr 0.01 --lr-stages 6000,12000,18000,24000 --margin 0.5 \
  --backbone-resume-root ../py-model/ResNet_50_Epoch_35.pth \
  --gpu-ids 0,1,2,3 \
  2>&1 | tee ../py-log/tri_hard_`date +'%m_%d-%H_%M'`.log  
