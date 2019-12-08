#export OMP_NUM_THREADS=4
  #--head-resume-root /data_luoqi/zhengmeisong/wkspace/py-model//RA_92/Head_Combine_Epoch_6_Batch_160368_Time_2019-12-07-15-29_checkpoint.pth \
	  #python myTrainDali.py  --backbone-name ${MODEL} --data-root /data_luoqi/zhengmeisong/data/gl2ms1m_img/ \
MODEL=RA_92
mkdir ../py-model/${MODEL}/
python myTrain.py  --backbone-name ${MODEL} --data-root /data_luoqi/zhengmeisong/data/gl2ms1m_img/ \
  --backbone-resume-root /data_luoqi/zhengmeisong/wkspace/py-model//RA_92/Backbone_RA_92_Epoch_6_Batch_160368_Time_2019-12-07-15-29_checkpoint.pth \
  --batch-size 128 --num-workers 4 \
  --lr 0.0001 --lr-stages 4,8,12 --num-epoch 15 \
  --loss-name Softmax --head-name Combine \
  --model-root ../py-model/${MODEL}/ \
  --gpu-ids 2,3 2>&1 | tee ../py-log/ra92Combine_`date +'%m_%d-%H_%M'`.log

