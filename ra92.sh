export OMP_NUM_THREADS=4
MODEL=RA_92
mkdir ../py-model/${MODEL}/
python myTrainDali.py  --backbone-name ${MODEL} --data-root /mnt/sdc/zhengmeisong/data/glintv2_emore_ms1m_img/ \
  --backbone-resume-root ../py-preTrain/Backbone_RA_92_Epoch_1_Batch_26728_Time_2019-12-03-23-01_checkpoint.pth \
  --head-resume-root ../py-preTrain/Head_Combine_Epoch_1_Batch_26728_Time_2019-12-03-23-01_checkpoint.pth \
  --batch-size 64 --num-workers 2 \
  --lr 0.0001 --lr-stages 4,8,12 --num-epoch 15 \
  --loss-name Softmax --head-name Combine \
  --model-root ../py-model/${MODEL}/ \
  --gpu-ids 0 2>&1 | tee ../py-log/ra92Combine_`date +'%m_%d-%H_%M'`.log

