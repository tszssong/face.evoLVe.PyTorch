#python trainDisstilling.py --data-root /data02/zhengmeisong/data/tmp/gl2ms1m_img/ \
python trainDisstilling.py --data-root /data02/zhengmeisong/data/gl2ms1m_img/ \
       --lr 0.1 --lr-stages 9,12,15 --num-epoch 20 \
       --backbone-name MobileV2 \
       --batch-size 128 \
       --gpu-ids 2,3 2>&1 | tee ../py-log/kd`date +'%m_%d-%H_%M'`.log  
