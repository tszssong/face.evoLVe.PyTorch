export OMP_NUM_THREADS=4
MODEL=RA_92
#python myTrain.py  --backbone-name $MODEL --data-root /cloud_data01/zhengmeisong/data/gl2ms1m_imgs/ \
python myTrain.py  --backbone-name ${MODEL} --data-root /data02/zhengmeisong/data/gl2ms1m_img/ \
  --batch-size 160 \
  --lr 0.1 --lr-stages 8,12,16,18 --num-epoch 25 \
  --loss-name Softmax --head-name Combine \
  --model-root ../py-model/${MODEL}/ \
  --gpu-ids 0,1 2>&1 | tee ../py-log/ra92Combine_`date +'%m_%d-%H_%M'`.log
