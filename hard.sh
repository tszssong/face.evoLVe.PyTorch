python tri_hard_train.py --data-root /data02/zhengmeisong/data/ms1m_align_112 \
  --num-workers 6 --num-epoch 1000\
  --bag-size 8192 --batch-size 1024 --test-freq 50 --save-freq 50\
  --lr 0.001 --lr-stages 2,5,8,12 --margin 0.3 \
  --backbone-resume-root ../py-model/ResNet_50_Epoch_35.pth \
  --gpu-ids 0,1,2,3 \
  2>&1 | tee ../py-log/tri_hard_`date +'%m_%d-%H_%M_%S'`.log
