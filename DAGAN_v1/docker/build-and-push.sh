#!/bin/bash -eux

# Run this script from the repo's root folder, pointing to ade20k for testing:
#
# $ ./docker/build-and-push.sh <ADEChallengeData2016_dir>

ade20k_dir="$(realpath "$1")"

# 1. Build Docker images for CPU and GPU

image="us-docker.pkg.dev/replicate/ha0tang/dagan_v1"
cpu_tag="$image:cpu"
gpu_tag="$image:gpu"
docker build -f docker/Dockerfile.cpu --tag "$cpu_tag" .
docker build -f docker/Dockerfile.gpu --tag "$gpu_tag" .

# 2. Test the Docker images on ade20k data

test_output_folder=/tmp/test-dagan/output

time docker run -it \
       -v "$ade20k_dir":/data \
       -v $test_output_folder/cpu:/results \
       $cpu_tag \
       --name GauGAN_DAGAN_ade_pretrained \
       --dataset_mode ade20k \
       --dataroot /data \
       --results_dir /results \
       --how_many 3

[ -f $test_output_folder/cpu/GauGAN_DAGAN_ade_pretrained/test_latest/images/synthesized_image/ADE_val_00000001.png ] || exit 1
[ -f $test_output_folder/cpu/GauGAN_DAGAN_ade_pretrained/test_latest/index.html ] || exit 1

time docker run --gpus all -it \
       -v "$ade20k_dir":/data \
       -v $test_output_folder/gpu:/results \
       $gpu_tag \
       --gpu_ids 0 \
       --name GauGAN_DAGAN_ade_pretrained \
       --dataset_mode ade20k \
       --dataroot /data \
       --results_dir /results \
       --how_many 3

[ -f $test_output_folder/gpu/GauGAN_DAGAN_ade_pretrained/test_latest/images/synthesized_image/ADE_val_00000001.png ] || exit 1
[ -f $test_output_folder/gpu/GauGAN_DAGAN_ade_pretrained/test_latest/index.html ] || exit 1

sudo rm -rf "$test_output_folder"

# 3. Push images

docker push $cpu_tag
docker push $gpu_tag
