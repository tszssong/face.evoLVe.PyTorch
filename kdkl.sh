       #--lr 0.1 --lr-stages 20,40,60,80,100,200,300,400 --num-epoch 500 \
       #--lr 0.1 --lr-stages 2,4,6,8,10,20,30,40 --num-epoch 500 \
export OMP_NUM_THREADS=4
python trainKL.py --data-root /cloud_data01/zhengmeisong/data/gl2ms1m_imgs/ \
       --lr 0.0001 --lr-stages 20,30,40 --num-epoch 50 \
       --backbone-resume-root ../py-preTrain/Backbone_MobileV2_Epoch_20_Batch_668200_Time_2019-11-30-15-49_checkpoint.pth \
       --head-resume-root ../py-preTrain/Head_Combine_Epoch_6_Batch_160368_Time_2019-11-30-06-36_checkpoint.pth \
       --teacher-resume-root ../py-preTrain/Backbone_RA_92_Epoch_6_Batch_160368_Time_2019-11-30-06-36_checkpoint.pth \
       --teacher-head-resume-root ../py-preTrain/Head_Combine_Epoch_6_Batch_160368_Time_2019-11-30-06-36_checkpoint.pth \
       --teacher-name RA_92 \
       --backbone-name MobileV2 \
       --head-name Combine \
       --teacher-head-name Combine \
       --batch-size 56 \
       --gpu-ids 2 2>&1 | tee ../py-log/kd`date +'%m_%d-%H_%M'`.log  
