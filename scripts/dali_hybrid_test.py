import argparse
import os
import shutil
import time
import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from timeit import default_timer as timer

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size,
                                              num_threads, device_id, prefetch_queue_depth=1, seed=12 + device_id)
        self.input = ops.FileReader(
            file_root=data_dir, shard_id=0, num_shards=1, random_shuffle=False)
        # let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode1 = ops.ImageDecoder(device=decoder_device, hybrid_huffman_threshold=100, output_type=types.RGB,
                                        device_memory_padding=device_memory_padding,
                                        host_memory_padding=host_memory_padding)
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, hybrid_huffman_threshold=100, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[
                                                     0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop,
                              resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.rrc = ops.RandomResizedCrop(device=dali_device, size=224)
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 *
                                                  255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        self.bri = ops.Brightness(device=dali_device)
        self.flip = ops.Flip(device=dali_device)
        self.crop = ops.Crop(crop=[32, 32], device=dali_device)

        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        # output = self.jpegs
        output = self.decode1(self.jpegs)
        output = self.res(output)
        # output = self.rrc(output)
        # output = self.cmnp(output)
        # output = output.gpu()
        # output = self.bri(output)
        # output = self.flip(output)
        output = self.crop(output)
        return [output, self.labels]


if __name__ == "__main__":
    traindir = "/mnt/optane-ssd/lipeng/imagenet/train"
    batch_size = 128
    num_threads = 8
    crop_size = 224
    dali_cpu = False
    epochs = 10
    iters = 1000
    pipe = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads, device_id=0,
                           data_dir=traindir, crop=crop_size, dali_cpu=dali_cpu)
    pipe.build()
    # train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / 1))

    pipe.run()  # run one

    cur_iter = 0
    t_start = timer()
    img = None
    for i in range(0, iters):
        pipe.run()

    print(type(img))

    t = timer() - t_start
    print("time: {:.3f} seconds".format(t))
    print("speed: {:.3f} images/second".format(iters * batch_size / t))
