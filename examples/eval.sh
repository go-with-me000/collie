#!/bin/bash

# 文件夹路径
folder_path="/mnt/petrelfs/chenkeyu1/program/collie/lora_weights/adaptor_07_11"

# 遍历文件夹，并获取子文件夹的名称
for dir_name in "$folder_path"/*; do
    if [ -d "$dir_name" ]; then
        # 运行命令，并将子文件夹的名称作为参数传递
        last_part=$(basename "$dir_name")
        mkdir -p logs/$last_part
        (srun -p llm --quotatype=spot --gres=gpu:1 python eval.py -a "$dir_name" 2>&1 | tee "logs/$last_part/log.txt") &
    fi
done

# 等待所有后台任务完成
wait
