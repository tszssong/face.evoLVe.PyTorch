python trainKL.py --data-root /data02/zhengmeisong/data/gl2ms1m_img/ \
       --lr 0.1 --lr-stages 200,300,400 --num-epoch 500 \
       --backbone-resume-root ../py-model/Backbone_MobileV2_Epoch_17_Batch_567970_Time_2019-11-29-01-52_checkpoint.pth \
       --head-resume-root ../py-preTrain/Head_Softmax_Epoch_15_Batch_915_Time_2019-11-29-10-38_checkpoint.pth \
       --teacher-resume-root ../py-preTrain/Backbone_RA_92_Epoch_1_Batch_26728_Time_2019-11-26-11-12_checkpoint.pth \
       --teacher-head-resume-root ../py-preTrain/Head_Softmax_Epoch_15_Batch_915_Time_2019-11-29-10-38_checkpoint.pth \
       --teacher-name RA_92 \
       --backbone-name MobileV2 \
       --head-name Softmax \
       --teacher-head-name Softmax \
       --batch-size 64 \
       --gpu-ids 0 2>&1 | tee ../py-log/kd`date +'%m_%d-%H_%M'`.log  


  #    --teacher-head-resume-root ../py-preTrain/Head_Combine_Epoch_1_Batch_26728_Time_2019-11-26-11-12_checkpoint.pth \
