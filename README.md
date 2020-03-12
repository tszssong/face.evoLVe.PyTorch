# pytorch 版商品识别 from ZhaoJ-face.evoLVe  
#### 环境  
pytorch 1.1.0 ~ 1.3.0 确定可行，其他版本只要能跑也可  
dali 参考[官方](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/installation.html)安装，最新版本会报警告，不影响使用  
其他pip install 安装  
#### 执行  
sh m2.sh  只要机器能访问/ai_data/应该可以直接跑，MobilenetV2训练  
提供[log文件](backup/MobileV2_03_12-16_29.log) 以供参考  
- 训练自己的数据：  
--data-root 改成自己的路径，--num-classes 改成自己路径下的类别数  
要注意的地方：  
1. 数据存放方式如下，可以修改myTrainDali.py第67行改为自己的路径  
   train_dir = os.path.join(args.data_root, 'data_100') #change data_100 to yourself subdir  
data_100  
|-class00  
|-class01  
|-......  
|-class99  
2. 每个类别放在同一个子路径下，路径下不宜出现图片之外的其他格式数据

####  其他  
识别是对齐后的112输入，用于商品识别需增加随机crop、旋转等操作，如改回224输入修改backbone第一层卷积stride=2  
