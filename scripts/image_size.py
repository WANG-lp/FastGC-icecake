from PIL import Image
from tqdm import tqdm
from operator import itemgetter

BASE_DIR = "/mnt/optane-ssd/lipeng/imagenet/"


def get_filenames(fname):
    flist = []
    with open(fname, 'r') as f:
        lines = f.readlines()
    for l in lines:
        flist.append(l.rstrip().split(" ")[0])
    return flist


def get_jpeg_size(fnames, fout):
    with open(fout, "w") as of:
        for fname in tqdm(fnames[:]):
            try:
               im = Image.open(BASE_DIR+fname)
               h, w= im.size
               of.write(fname)
               of.write(" ")
               of.write("{} {}\n".format(h,w))
            except:
                of.write(fname)
                of.write(" NA\n")

def parse_log():
    logf = "finfo_size.txt"
    images = []
    with open(logf, "r") as f:
        lines = f.readlines()
    for l in lines[:]:
        ls = l.rstrip().split(" ")
        images.append({"name":ls[0], "height":int(ls[1]), "width":int(ls[2])})
    newlist = sorted(images, key=lambda k: k['height']*k['width'], reverse=True)
    print(newlist[int(len(newlist)/2)])
    print(newlist[:10])
if __name__ == "__main__":
    flist = get_filenames("/mnt/optane-ssd/lipeng/imagenet/train.txt")
    # get_jpeg_size(flist, "finfo_size.txt")
    parse_log()