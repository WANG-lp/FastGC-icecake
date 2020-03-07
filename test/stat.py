import numpy as np
import os
import PIL
import PIL.Image
from tqdm import tqdm

total_size = 0
numpy_size = 0
def convert_image_to_dltensor(image_folder):
    global total_size, numpy_size
    files = []
    with open(image_folder + 'train.txt', 'r') as f:
        files = [line.strip() for line in f if line is not '']
    with open("np_raw_size.txt", 'w+') as f:
        for l in tqdm(files):
            filename = image_folder+l
            raw_size = os.path.getsize(filename)
            total_size += raw_size
            npbuff = np.asarray(PIL.Image.open(filename), dtype=np.uint8)
            numpy_size += npbuff.size
            f.write("filename:{} raw:{} -> np:{}\n".format(l, raw_size, npbuff.size))
            # f.flush()
            # break

if __name__=="__main__":
    convert_image_to_dltensor("/mnt/optane-ssd/lipeng/imagenet/")
    print("raw image size: {}".format(total_size))
    print("numpy array size: {}".format(numpy_size))

    