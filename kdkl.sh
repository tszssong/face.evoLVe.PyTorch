python trainL2.py --data-root /home/zhengmeisong/data/glintv2_ms1m_img/ \
       --lr 0.1 --lr-stages 4,9,12,15 --num-epoch 20 \
       --backbone-resume-root ../py-preTrain/Backbone_MobileV2_Epoch_17_Batch_567970_Time_2019-11-29-01-52_checkpoint.pth \
       --head-resume-root ../py-preTrain/Head_Softmax_Epoch_15_Batch_915_Time_2019-11-29-10-38_checkpoint.pth \
       --teacher-resume-root ../py-preTrain/Backbone_RA_92_Epoch_1_Batch_26728_Time_2019-11-26-11-12_checkpoint.pth \
       --teacher-head-resume-root ../py-preTrain/Head_Softmax_Epoch_15_Batch_915_Time_2019-11-29-10-38_checkpoint.pth \
       --teacher-name RA_92 \
       --backbone-name MobileV2 \
       --head-name Softmax \
       --teacher-head-name Softmax \
       --batch-size 60 \
       --gpu-ids 6,7 2>&1 | tee ../py-log/kd`date +'%m_%d-%H_%M'`.log  


  #    --teacher-head-resume-root ../py-preTrain/Head_Combine_Epoch_1_Batch_26728_Time_2019-11-26-11-12_checkpoint.pth \