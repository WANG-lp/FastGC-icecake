import os
import time
from tqdm import tqdm
import random
import string

buf4 = str('a' * 4096)
buf8 = str('a' * 4096 * 2)
buf16 = str('a' * 4096 * 4)
buf32 = str('a' * 4096 * 8)
buf64 = str('a' * 4096 * 16)
buf128 = str('a' * 4096 * 32)
buf256 = str('a' * 4096 * 64)
buf512 = str('a' * 4096 * 128)
buf1024 = str('a' * 4096 * 256)

bufs = [buf4, buf8, buf16, buf32, buf64, buf128, buf256, buf512, buf1024]

def create_and_count_time(filepath):
    global bufs
    timelist = []
    for buf in bufs:
        randomName = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k = 8))
        # print(randomName)
        start_t = time.perf_counter()
        with open(randomName, 'w+') as f:
            f.write(buf)
        os.remove(randomName)
        end_t = time.perf_counter()
        timelist.append(end_t - start_t)
    return timelist

if __name__=="__main__":
    with open("metadata-mon.txt", "w+") as f:
        for i in tqdm(range(3)):
            t1= create_and_count_time("/tmp/a.txt")
            f.write(str(t1)+"\n")
            f.flush()
            time.sleep(1)
            


