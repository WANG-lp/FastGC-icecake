import os,sys
from timeit import default_timer as timer
import random
import numpy as np

iters = 0
filenames = []
ratio = 0.1

batch_size = 256

hdd_path = "/home/lwangay/datasets/imagenet/"
nvme_path = "/mnt/optane-ssd/lipeng/imagenet/"
memory_path = "/dev/shm/imagenet/"
file_list = "/mnt/optane-ssd/lipeng/imagenet/train.txt"
def read_file_list(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    for l in lines:
        filenames.append(l.split(' ')[0].strip())
    random.shuffle(filenames)
    print("total file: {}".format(len(filenames)))
def read_bench():
    idx = 0
    total_len = 0
    per_buff = []
    indicies = np.arange(0,len(filenames),1)
    np.random.shuffle(indicies)
    for i in range(iters * batch_size):
        with open(memory_path+filenames[idx], 'rb') as f:
            per_buff.append(f.read())
    with open("recv-ratio-log-{}.txt".format(ratio), "w+") as of:
        for it in range(0, iters):
            cached_pos = (it*1.0/ratio) * len(filenames)
            miss_count = 0
            time_s = timer()
            for b in range(batch_size):
                if indicies[idx] < cached_pos:
                    pass
                else:
                    miss_count +=1
                idx +=1
            t = timer() - time_s
            of.write("{}\n".format((miss_count*1.0)/batch_size))
            of.flush()

if __name__=="__main__":
    iters = int(sys.argv[1])
    ratio = float(sys.argv[2])
    print("iters:{}, recv time:{}".format(iters, ratio))
    read_file_list(file_list)
    read_bench()