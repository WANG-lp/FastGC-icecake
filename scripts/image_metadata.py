import PIL
from PIL import Image, ExifTags
import subprocess
BASE_DIR = "/mnt/optane-ssd/lipeng/imagenet/"


def get_filenames(fname):
    flist = []
    with open(fname, 'r') as f:
        lines = f.readlines()
    for l in lines:
        flist.append(l.rstrip().split(" ")[0])
    return flist


def get_jpeg_meta(fnames, fout):
    with open(fout, "w") as of:
        for fname in fnames[:]:
            try:
                # result = subprocess.run(['ls', '-l'], stdout=subprocess.PIPE)
                result = subprocess.run(
                    ['identify', '-verbose', BASE_DIR+fname], stdout=subprocess.PIPE)
                result_str = result.stdout.decode().split("\n")
                # print(result_str)
                for l in result_str:
                    ls = l.strip(" ").strip("\t")
                    if ls.startswith("jpeg:sampling-factor:"):
                        of.write(fname)
                        of.write(' ')
                        of.write(ls)
                        of.write('\n')
            except:
                of.write(fname)
                of.write(" NA\n")


if __name__ == "__main__":
    flist = get_filenames("/mnt/optane-ssd/lipeng/imagenet/train.txt")
    get_jpeg_meta(flist, "finfo.txt")
