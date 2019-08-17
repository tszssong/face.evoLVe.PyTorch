python tri_hard_train.py --data-root /data3/zhengmeisong/data/ms1m_emore_img \
  --num-workers 20 --num-epoch 1000\
  --bag-size 3000 --batch-size 600 --test-bag 50 \
  --lr 0.001 --margin 0.3\
  --backbone-resume-root ../py-model/ResNet_50_Epoch_35.pth \
  --gpu-ids 0,1,2,3 \
  2>&1 | tee ../py-log/tri_hard_`date +'%m_%d-%H_%M_%S'`.log
