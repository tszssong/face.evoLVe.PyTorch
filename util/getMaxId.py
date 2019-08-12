import sys, os
last_id = 0
max_num = 0
max_id = 0
count = 0
id_dict = {}
with open('/home/ubuntu/zms/data/ms1m_emore_img/imgs.lst') as f:
  lines = f.readlines()
  for line in lines:
    path, id = line.strip().split(' ')
    id = int(id)
    if not id in id_dict:
      id_dict[id] = 1
    else:
      id_dict[id] += 1
    if(last_id==id):
      count += 1
    else:
      if(count>max_num):
        max_num = count
        max_id = last_id
        print("max id:%d, count:%d"%(max_id, max_num))
      count = 0
      last_id = id
# print(dict)
smallcount = 0
for key, value in id_dict.items():
  if(value<=3):
    print( "%d - %d"%(key,value) )
    smallcount += 3
print("max id:%d, count:%d"%(max_id, max_num))
print("img less than 3 ids: %d"%smallcount)

