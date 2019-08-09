python tri_train.py --data-root /data3/zhengmeisong/data/ms1m_align_112 \
  --num-workers 16 --num-epoch 1250000\
  --bag-size 1200 --batch-size 120 --test_epoch 50 \
  2>&1 | tee ../py-log/lr01_b40`date +'%m_%d-%H_%M_%S'`.log
