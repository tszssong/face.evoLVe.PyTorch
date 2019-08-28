import sys, os, shutil
if not os.path.isdir('./JPEGImages'):
    os.makedirs('./JPEGImages')
path = sys.argv[1]
print(path)
for root, dirs, files in os.walk(path):
    print(root, dirs, files)
    for dir in dirs:
        for subroot, subdirs, subfiles in os.walk(path + '/' + dir + '/'):
            for file in subfiles:
                if'._' in file:
                    continue
                if '.jpg' in file:
                    shutil.copy(path + '/' + dir + '/' + file, './JPEGImages/'+file)
