#python tri_hard_train.py --data-root /data3/zhengmeisong/lmdbTrain/gl2ms1m_dl23W1f1_150WW1/ \
python tri_hard_train.py --data-root /home/zhengmeisong/data/gl2ms1m_dl23W1f1_150WW1_img/ \
  --num-workers 0 --num-epoch 100 \
  --bag-size 72000 --batch-size 120 --test-freq 10 --save-freq 50\
  --lr 0.01 --lr-stages 20,400,1800 --margin 0.5 \
  --backbone-name IR_SE_152 \
  --backbone-resume-root ../py-preTrain/IR_SE_152_Epoch_16.pth \
  --gpu-ids 0,1,2,3,6,7 \
  2>&1 | tee ../py-log/tri_irse152_`date +'%m_%d-%H_%M'`.log  
