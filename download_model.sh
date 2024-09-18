#!/bin/bash

# 模型标识符（模型名称），例如：bert-base-uncased
MODEL_NAME="bert-base-uncased"

# 本地目录路径，替换为你希望保存模型的本地路径
LOCAL_DIR="/home/lucy/user/models/bert-base-uncased"

# 设置Hugging Face镜像站环境变量（可选，用于加速下载）
# 取消下行注释并替换为实际的镜像站地址
# export HF_ENDPOINT="https://hf-mirror.com"

# 使用huggingface-cli下载模型
huggingface-cli download \
  --resume-download \
  "$MODEL_NAME" \
  --local-dir "$LOCAL_DIR"

# 检查模型是否下载成功
if [ -d "$LOCAL_DIR" ]; then
  echo "模型已成功下载到 $LOCAL_DIR"
else
  echo "模型下载失败"
fi

