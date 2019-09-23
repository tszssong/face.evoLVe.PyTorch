python tri_hard_train.py --data-root /data3/zhengmeisong/data/gl2ms1m_dl23W1f1_150WW1_img \
  --num-workers 16 --num-epoch 100 \
  --bag-size 20000 --batch-size 1600 --test-freq 100 --save-freq 50\
  --lr 0.01 --lr-stages 6000,12000,18000,24000 --margin 0.5 \
  --backbone-resume-root ../py-model/ResNet_50_Epoch_35.pth \
  --gpu-ids 0,1,2,3 \
  2>&1 | tee ../py-log/tri_hard_`date +'%m_%d-%H_%M'`.log
