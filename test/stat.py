from pyicecake import pyicecake
import numpy as np
import os
import PIL
import PIL.Image
import cupy as cp
from tqdm import tqdm

total_size = 0
numpy_size = 0
def convert_image_to_dltensor(image_folder):
    global total_size, numpy_size
    files = []
    with open(image_folder + 'train.txt', 'r') as f:
        files = [line.strip() for line in f if line is not '']
    for l in tqdm(files):
        filename = image_folder+l
        total_size += os.path.getsize(filename)
        npbuff = np.asarray(PIL.Image.open(filename), dtype=np.uint8)
        numpy_size += npbuff.size
        cpbuff = cp.asarray(npbuff)
        dltensor = cpbuff.toDlpack()
        gc.put_dltensor(filename, dltensor)
        # break

if __name__=="__main__":
    gc = pyicecake.GPUCache(4*1024*1024*1024) # 4GB
    print(gc.get_write_size())
    convert_image_to_dltensor("/mnt/optane-ssd/lipeng/imagenet/")
    print("tensor size: {}".format(gc.get_write_size()))
    print("raw image size: {}".format(total_size))
    print("numpy array size: {}".format(numpy_size))

    