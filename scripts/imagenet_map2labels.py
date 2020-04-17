import os

train_file_list = "/mnt/optane-ssd/lipeng/imagenet/train.txt"
label_list = "/mnt/optane-ssd/lipeng/imagenet/map_clsloc.txt"
output_file_list = "/mnt/optane-ssd/lipeng/imagenet/train_label.txt"


def map2label(file_list, label_list, output):
    lines = []
    labels = dict()
    with open(file_list, "r") as f:
        lines = f.readlines()
    with open(label_list, "r") as f:
        ll = f.readlines()
        for l in ll:
            folder_name, idx, common_name = l.strip().split(" ")
            labels[folder_name] = {"idx": idx, "common_name": common_name}
    with open(output, "w+") as outf:
        for l in lines:
            folder_name = l.strip().split("/")[1]
            outline = "{} {}".format(
                l.strip(), labels[folder_name]['idx'])
            outf.write(outline+"\n")


if __name__ == "__main__":
    map2label(train_file_list, label_list, output_file_list)
