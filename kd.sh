python trainDisstilling.py --data-root /mnt/sdc/zhengmeisong/data/glintv2_emore_ms1m_img/ \
       --lr 0.001 --lr-stages 4,9,12 --num-epoch 15 \
       --backbone-resume-root ../py-model/Backbone_MobileV2_Epoch_4_Batch_171060_Time_2019-11-25-06-11_checkpoint.pth \
       --head-resume-root ../py-model/Head_ArcFace_Epoch_4_Batch_171060_Time_2019-11-25-06-11_checkpoint.pth \
       --backbone-name MobileV2 \
       --head-name ArcFace \
       --batch-size 200 \
       --gpu-ids 0,1 2>&1 | tee ../py-log/kd`date +'%m_%d-%H_%M'`.log  
