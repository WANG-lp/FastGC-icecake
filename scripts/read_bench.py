import os,sys
from timeit import default_timer as timer
import random

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
    for i in range(iters * batch_size):
        with open(memory_path+filenames[idx], 'rb') as f:
            per_buff.append(f.read())
    with open("nvme-time-log-{}.txt".format(ratio), "w+") as of:
        for it in range(0, iters):
            time_s = timer()
            for b in range(batch_size):
                buff = None
                if random.random() > ratio:
                    buff = per_buff[idx]
                else:
                    with open(nvme_path+filenames[idx], 'rb') as f:
                        buff = f.read()
                length = len(buff)                
                total_len += length
                idx +=1
            t = timer() - time_s
            of.write("{}\n".format(t))
            of.flush()

if __name__=="__main__":
    iters = int(sys.argv[1])
    ratio = float(sys.argv[2])
    print("iters:{}, ratio:{}".format(iters, ratio))
    read_file_list(file_list)
    read_bench()