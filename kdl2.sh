python trainL2.py --data-root /home/zhengmeisong/data/glintv2_ms1m_img/ \
       --lr 0.01 --lr-stages 4,9,12 --num-epoch 15 \
       --backbone-resume-root ../py-model/Backbone_MobileV2_Epoch_4_Batch_171060_Time_2019-11-25-06-11_checkpoint.pth \
       --head-resume-root ../py-model/Head_ArcFace_Epoch_4_Batch_171060_Time_2019-11-25-06-11_checkpoint.pth \
       --teacher-resume-root ../py-preTrain/Backbone_RA_92_Epoch_1_Batch_26728_Time_2019-11-26-11-12_checkpoint.pth \
    #    --teacher-head-resume-root ../py-preTrain/Head_Combine_Epoch_1_Batch_26728_Time_2019-11-26-11-12_checkpoint.pth \
       --teacher-head-resume-root ../py-model/Head_Combine_Epoch_15_Batch_915_Time_2019-11-28-19-33_checkpoint.pth \
       --teacher-name RA_92 \
       --backbone-name MobileV2 \
       --head-name Combine \
       --teacher-head-name Combine \
       --batch-size 6 \
       --gpu-ids 6,7 2>&1 | tee ../py-log/kd`date +'%m_%d-%H_%M'`.log  
