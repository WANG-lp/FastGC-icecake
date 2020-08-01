# export PUT=1
export MAX_ITER=1000
export BASEDIR="/mnt/optane-ssd/lipeng/imagenet/"

PREFIX="predict_f3"

MODEL="alexnet"

# for i in 2 4 8 16 32 64 128 256; do 
# PIPE=0 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_jcache_batch_${i}.log
# done

# for i in 2 4 8 16 32 64 128 256; do 
# PIPE=1 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_diesel_batch_${i}.log
# done


# for i in 2 4 8 16 32 64 128 256; do 
# PIPE=2 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_nvjpeg_batch_${i}.log
# done

# for i in 2 4 8 16 32 64 128 256; do 
# PIPE=2 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} --dali_cpu /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_cpu_batch_${i}.log
# done


for i in 2 4 8 16 32 64 128 256; do 
PIPE=3 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_nvjpeg_memory_batch_${i}.log
done



MODEL="resnet18"

# for i in 2 4 8 16 32 64 128 256; do 
# PIPE=0 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_jcache_batch_${i}.log
# done

# for i in 2 4 8 16 32 64 128 256; do 
# PIPE=1 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_diesel_batch_${i}.log
# done


# for i in 2 4 8 16 32 64 128 256; do 
# PIPE=2 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_nvjpeg_batch_${i}.log
# done

# for i in 2 4 8 16 32 64 128 256; do 
# PIPE=2 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} --dali_cpu /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_cpu_batch_${i}.log
# done

for i in 2 4 8 16 32 64 128 256; do 
PIPE=3 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_nvjpeg_memory_batch_${i}.log
done



# MODEL="resnet50"

# # for i in 2 4 8 16 32 64 128 256; do 
# # PIPE=0 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_jcache_batch_${i}.log
# # done

# # for i in 2 4 8 16 32 64 128 256; do 
# # PIPE=1 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_diesel_batch_${i}.log
# # done


# # for i in 2 4 8 16 32 64 128 256; do 
# # PIPE=2 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_nvjpeg_batch_${i}.log
# # done

# # for i in 2 4 8 16 32 64 128 256; do 
# # PIPE=2 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} --dali_cpu /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_cpu_batch_${i}.log
# # done

# for i in 2 4 8 16 32 64 128 256; do 
# PIPE=3 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_nvjpeg_memory_batch_${i}.log
# done


MODEL="shufflenet_v2_x1_0"

# for i in 2 4 8 16 32 64 128 256; do 
# PIPE=0 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_jcache_batch_${i}.log
# done

# for i in 2 4 8 16 32 64 128 256; do 
# PIPE=1 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_diesel_batch_${i}.log
# done


# for i in 2 4 8 16 32 64 128 256; do 
# PIPE=2 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_nvjpeg_batch_${i}.log
# done

# for i in 2 4 8 16 32 64 128 256; do 
# PIPE=2 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} --dali_cpu /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_cpu_batch_${i}.log
# done
for i in 2 4 8 16 32 64 128 256; do 
PIPE=3 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_nvjpeg_memory_batch_${i}.log
done



MODEL="mobilenet_v2"
# for i in 2 4 8 16 32 64 128 256; do 
# PIPE=0 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_jcache_batch_${i}.log
# done

# for i in 2 4 8 16 32 64 128 256; do 
# PIPE=1 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_diesel_batch_${i}.log
# done


# for i in 2 4 8 16 32 64 128 256; do 
# PIPE=2 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_nvjpeg_batch_${i}.log
# done

# for i in 2 4 8 16 32 64 128 256; do 
# PIPE=2 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} --dali_cpu /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_cpu_batch_${i}.log
# done

for i in 2 4 8 16 32 64 128 256; do 
PIPE=3 python main.py --opt-level O3 --epoch 1 -b ${i}  --workers 4 -a ${MODEL} /mnt/optane-ssd/lipeng/imagenet/ | tee ${PREFIX}_${MODEL}_nvjpeg_memory_batch_${i}.log
done
