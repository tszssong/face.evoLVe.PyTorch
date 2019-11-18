python trainDisstilling.py --data-root /home/zhengmeisong/data1/ms1m_emore_imgs \
       --lr 0.1 --lr-stages 9,12,15 --num-epoch 20 \
       --backbone-name MobileV2 \
       --batch-size 128 \
       --gpu-ids 2,3 2>&1 | tee ../py-log/kd`date +'%m_%d-%H_%M'`.log  
