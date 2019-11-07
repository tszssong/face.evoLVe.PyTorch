DATAROOT=/cloud_data01/zhengmeisong/TestData/JA-Test/
MODEL=/cloud_data01/zhengmeisong/wkspace/olx/py-model/MV2/MV2.pth
python saveFeatures.py --backbone-resume-root $MODEL --backbone-name MobileV2 \
        --data-root $DATAROOT --list imgs20.lst --gpu-ids 2
