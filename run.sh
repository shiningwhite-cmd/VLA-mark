#!/usr/bin/env bash
# 说明：该脚本从环境变量 MODEL_PATH 读取模型路径，并作为 --model_path 传入 all_mark.py
# 使用示例：
#   export MODEL_PATH=/abs/path/to/your/model
#   bash all_mark.sh
# 若未设置 MODEL_PATH，将给出提示并退出。

if [ -z "${MODEL_PATH}" ]; then
  echo "[ERROR] 未检测到环境变量 MODEL_PATH，请先执行：export MODEL_PATH=/your/model/path"
  exit 1
fi

python3 main.py \
    --json_path ./dataset/amber/query_generative.json \
    --image_dir ./dataset/amber/image \
    --model_path "${MODEL_PATH}" \
    --range_num 10 \
    --model_name llava \
    --task_name AMBER \
    --data_suffix .jpg \
    --similarity_scheme cosine \
    --max_tokens 100 \
    --min_tokens 90
