import PIL
from PIL import Image, ExifTags
import subprocess
import os
BASE_DIR = "/mnt/optane-ssd/lipeng/imagenet/"


def get_filenames(fname):
    flist = []
    with open(fname, 'r') as f:
        lines = f.readlines()
    for l in lines:
        flist.append(l.rstrip().split(" ")[0])
    return flist


def get_file_size(fname):
    size = os.path.getsize(fname)
    if size:
        return size
    else:
        return -1


def get_jpeg_meta(fnames, fout):
    with open(fout, "w") as of:
        for fname in fnames[:]:
            try:
                # result = subprocess.run(['ls', '-l'], stdout=subprocess.PIPE)
                result = subprocess.run(
                    ['/home/lwangay/anaconda3/bin/jpegtran', '-crop', '224x224+0+0', BASE_DIR+fname], stdout=subprocess.PIPE)
                new_len = len(result.stdout)
                # print(result.stdout[:10])
                # print(result_str)
                # break
                # break
                old_len = get_file_size(BASE_DIR+fname)
                of.write("{}: {} -> {}".format(fname, old_len, new_len))
                of.write("\n")
                with open(BASE_DIR+fname, "wb") as f:
                    f.write(result.stdout)
                # break
                # for l in result_str:
                #     ls = l.strip(" ").strip("\t")
                #     #if ls.startswith("jpeg:sampling-factor:"):
                #     of.write(fname)
                #     of.write('\n')
                #     of.write(ls)
                #     of.write('\n')
            except:
                of.write(fname)
                of.write(" NA\n")


def show_reduced(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    old_size = 0
    new_size = 0
    for l in lines:
        ll = l.rstrip().split(" ")
        old_size += int(ll[1])
        new_size += int(ll[3])
    print(old_size)
    print(new_size)
    print("old ave: {}".format(old_size/len(lines)))
    print("new ave: {}".format(new_size/len(lines)))

def copyfiles(flist):
    from shutil import copyfile

    with open(flist, "r") as fl:
        fnames = fl.readlines()
    for f in fnames:
        f = f.strip()
        os.makedirs(os.path.dirname(BASE_DIR+"cropped/"+f), exist_ok=True)
        copyfile(BASE_DIR+f, BASE_DIR+"cropped/"+f)

def crop(BASEDIR, flist):
    with open(flist, "r") as fl:
        fnames = fl.readlines()
    for f in fnames:
        f = f.strip()
        result = subprocess.run(
                        ['/home/lwangay/anaconda3/bin/jpegtran', '-crop', '224x224+0+0', BASEDIR+f], stdout=subprocess.PIPE)
        new_len = len(result.stdout)
        with open(BASEDIR+f, "wb") as f:
            f.write(result.stdout)
        
if __name__ == "__main__":
    # flist = get_filenames("/mnt/optane-ssd/lipeng/imagenet/train.txt")
    # get_jpeg_meta(flist, "finfo.txt")
    # show_reduced("./file_crop.txt")
    # copyfiles("/mnt/optane-ssd/lipeng/imagenet/image_ok_list.txt")

    crop("/mnt/optane-ssd/lipeng/imagenet/cropped/","/mnt/optane-ssd/lipeng/imagenet/image_ok_list.txt")
