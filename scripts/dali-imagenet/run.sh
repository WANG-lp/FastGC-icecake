export PUT=1

export BASEDIR="/mnt/optane-ssd/lipeng/imagenet/cropped/"
PIPE=0 python main.py --opt-level O1 --epoch 1 -a alexnet --dali_cpu /mnt/nvme-ssd/lipeng/imagenet | tee alexnet_diesel_roi_cpu.log

export BASEDIR="/mnt/optane-ssd/lipeng/imagenet/"
PIPE=1 python main.py --opt-level O1 --epoch 1 -a alexnet --dali_cpu /mnt/nvme-ssd/lipeng/imagenet | tee alexnet_diesel_cache_only_cpu.log

sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
PIPE=2 python main.py --opt-level O1 --epoch 1 -a alexnet --dali_cpu /mnt/nvme-ssd/lipeng/imagenet | tee alexnet_lustre_cpu.log





export BASEDIR="/mnt/optane-ssd/lipeng/imagenet/cropped/"
PIPE=0 python main.py --opt-level O1 --epoch 1 -a mobilenet_v2 --dali_cpu /mnt/nvme-ssd/lipeng/imagenet | tee mobilenet_diesel_roi_cpu.log

export BASEDIR="/mnt/optane-ssd/lipeng/imagenet/"
PIPE=1 python main.py --opt-level O1 --epoch 1 -a mobilenet_v2 --dali_cpu /mnt/nvme-ssd/lipeng/imagenet | tee mobilenet_diesel_cache_only_cpu.log

sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
PIPE=2 python main.py --opt-level O1 --epoch 1 -a mobilenet_v2 --dali_cpu /mnt/nvme-ssd/lipeng/imagenet | tee mobilenet_lustre_cpu.log




export BASEDIR="/mnt/optane-ssd/lipeng/imagenet/cropped/"
PIPE=0 python main.py --opt-level O1 --epoch 1 -a shufflenet_v2_x1_0 --dali_cpu /mnt/nvme-ssd/lipeng/imagenet | tee shufflenet_diesel_roi_cpu.log

export BASEDIR="/mnt/optane-ssd/lipeng/imagenet/"
PIPE=1 python main.py --opt-level O1 --epoch 1 -a shufflenet_v2_x1_0 --dali_cpu /mnt/nvme-ssd/lipeng/imagenet | tee shufflenet_diesel_cache_only_cpu.log

sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
PIPE=2 python main.py --opt-level O1 --epoch 1 -a shufflenet_v2_x1_0 --dali_cpu /mnt/nvme-ssd/lipeng/imagenet | tee shufflenet_lustre_cpu.log




export BASEDIR="/mnt/optane-ssd/lipeng/imagenet/cropped/"
PIPE=0 python main.py --opt-level O1 --epoch 1 -a resnet18  --dali_cpu /mnt/nvme-ssd/lipeng/imagenet | tee resnet18_diesel_roi_cpu.log

export BASEDIR="/mnt/optane-ssd/lipeng/imagenet/"
PIPE=1 python main.py --opt-level O1 --epoch 1 -a resnet18  --dali_cpu /mnt/nvme-ssd/lipeng/imagenet | tee resnet18_diesel_cache_only_cpu.log

sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
PIPE=2 python main.py --opt-level O1 --epoch 1 -a resnet18  --dali_cpu /mnt/nvme-ssd/lipeng/imagenet | tee resnet18_lustre_cpu.log





export BASEDIR="/mnt/optane-ssd/lipeng/imagenet/cropped/"
PIPE=0 python main.py --opt-level O1 --epoch 1 -a resnet50  --dali_cpu /mnt/nvme-ssd/lipeng/imagenet | tee resnet50_diesel_roi_cpu.log

export BASEDIR="/mnt/optane-ssd/lipeng/imagenet/"
PIPE=1 python main.py --opt-level O1 --epoch 1 -a resnet50  --dali_cpu /mnt/nvme-ssd/lipeng/imagenet | tee resnet50_diesel_cache_only_cpu.log

sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
PIPE=2 python main.py --opt-level O1 --epoch 1 -a resnet50  --dali_cpu /mnt/nvme-ssd/lipeng/imagenet | tee resnet50_lustre_cpu.log