#python trainDisstilling.py --data-root /data02/zhengmeisong/data/tmp/gl2ms1m_img/ \
export OMP_NUM_THREADS=4

python trainDisstilling.py --data-root /home/zhengmeisong/hpc37/gl2ms1m_img/ \
       --lr 0.1 --lr-stages 9,12,15,18 --num-epoch 20 \
       --backbone-name MobileV2 \
       --batch-size 128 --head-name Combine \
       --gpu-ids 0,1 2>&1 | tee ../py-log/kd`date +'%m_%d-%H_%M'`.log  
