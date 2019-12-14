id_dict = {}
with open('imgs.lst', 'r') as fp:
  lines = fp.readlines()
  for line in lines:
    path, label = line.strip().split(' ')
    if not label in id_dict:
      id_dict[label] = [path]
    else:
      id_dict[label].append(path)
#print(id_dict)
sort_dict = {}
for key, value in id_dict.iteritems():
#  print(key, len(value))
  if not len(value) in sort_dict:
    sort_dict[len(value)] = 1
  else:
    sort_dict[len(value)] += 1
  if len(value) < 5:
    with open('lessThan5.lst', 'a') as fw:
      for path in value:
        fw.write(path+'\n')
    with open('lessThan5dirs.lst', 'a') as fw:
      fw.write(value[0].split('/')[-2]+'\n')
print(sort_dict)
