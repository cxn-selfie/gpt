

#!/bin/bash

# 检查是否安装了huggingface-cli
if ! command -v huggingface-cli &> /dev/null
then
    echo "huggingface-cli 未安装。正在安装..."
    pip install huggingface_hub
fi

# 设置模型名称
MODEL_NAME="THUDM/glm-4-9b-chat"

# 使用huggingface-cli下载模型

export HF_ENDPOINT=https://hf-mirror.com

echo "正在从Hugging Face下载模型: $MODEL_NAME"
huggingface-cli download $MODEL_NAME --local-dir /home/lucy/user/models/glm-4-9b-chat

# 检查下载是否成功
if [ $? -eq 0 ]; then
    echo "模型下载成功。"
else
    echo "模型下载失败。可能的原因包括："
    echo "1. 网络连接问题"
    echo "2. 存储空间不足"
    echo "3. 权限问题"
    echo "4. Hugging Face服务器问题"
    echo "请检查以上可能的原因并重试。"
fi



