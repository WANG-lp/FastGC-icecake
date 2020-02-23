#!/usr/bin/env python3

import os.path
import fnmatch

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline

def list_files(image_dir):
    flist = []
    for root, dir, files in os.walk(image_dir):
        for items in fnmatch.filter(files, "*.JPEG"):
            flist.append(root+'/'+items)
    return flist

image_dir = "/mnt/optane-ssd/lipeng/imagenet/val"
batch_size = 4

class SimplePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(SimplePipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = image_dir)
        # instead of path to file directory file with pairs image_name image_label_value can be provided
        # self.input = ops.FileReader(file_root = image_dir, file_list = image_dir + '/file_list.txt')
        self.decode = ops.ImageDecoder(device = 'cpu', output_type = types.RGB)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        return (images, labels)

if __name__=="__main__":
    pipe = SimplePipeline(batch_size, 1, 0)
    pipe.build()
    for i in range(0,10):
        pipe_out = pipe.run()
        images, labels = pipe_out
        print(images)
        print(labels)