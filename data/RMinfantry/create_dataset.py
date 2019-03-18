import os
import random

# 3 parameters, can be set
JPGfile_dir = '/home/kangyi/data/RMinfantry/JPEGImages'
TXTfile_dir = '/home/kangyi/data/RMinfantry/ImageSets/Main'
trainval_ratio = 0.85

# push all names of images into a list
list_all = []
for root, dirs, files in os.walk(JPGfile_dir):
    for file in files:
        if os.path.splitext(file)[1] == '.jpg':
            list_all.append(os.path.splitext(file)[0])
print("image set is {} in total.".format(list_all.__len__()))

# pick up some for training and others for testing randomly
trainval_size = (int)(round(list_all.__len__() * trainval_ratio))
test_size = list_all.__len__() - trainval_size
print("trainval set is {} in total.".format(trainval_size))
print("test set is {} in total.".format(test_size))
all_index = range(list_all.__len__())
trainval_index = random.sample(all_index, trainval_size)
test_index = [x for x in all_index if x not in trainval_index]

# build trainval.txt
with open(TXTfile_dir+'/trainval.txt', 'w') as f:
    for i in trainval_index:
        f.write(list_all[i])
        f.write('\n')
f.close()

# build test.txt
with open(TXTfile_dir+'/test.txt', 'w') as f:
    for i in test_index:
        f.write(list_all[i])
        f.write('\n')
f.close()
