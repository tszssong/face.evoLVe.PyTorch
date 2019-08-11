import sys, os
last_id = 0
max_num = 0
max_id = 0
count = 0
with open('imgs.lst') as f:
  lines = f.readlines()
  for line in lines:
    path, id = line.strip().split(' ')
    id = int(id)
    if(last_id==id):
      count += 1
    else:
      if(count>max_num):
        max_num = count
        max_id = last_id
        print("max id:%d, count:%d"%(max_id, max_num))
      count = 0
      last_id = id
       
print("max id:%d, count:%d"%(max_id, max_num))
