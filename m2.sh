python myTrain.py --data-root /data02/zhengmeisong/data/tmp/gl2ms1m_img/ \
       --head-name Combine \
       --loss-name Softmax \
       --batch-size 256 --num-epoch 30 \
       --lr 0.1 --lr-stages 9,15,20,24 \
       --gpu-ids 3 2>&1 | tee ../py-log/m2_combine_`date +'%m_%d-%H_%M'`.log 
