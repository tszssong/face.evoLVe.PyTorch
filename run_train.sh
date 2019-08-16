python tri_train.py --data-root /home/zhengmeisong/ol11/data/ms1m_emore_img \
  --num-workers 16 --num-epoch 120000\
  --bag-size 8192 --batch-size 128 --test-epoch 2000 \
  --num-loaders 4 --gpu-ids 0,1,2,3,4,5,6,7 \
  2>&1 | tee ../py-log/tri_hard_`date +'%m_%d-%H_%M_%S'`.log
