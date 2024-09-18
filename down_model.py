################################
###### 下载单个文件
################################
# from huggingface_hub import hf_hub_download
# #
# # hf_hub_download(repo_id="RWKV/v5-EagleX-v2-7B-pth",local_dir='/home/zp/下载/rwkv',filename="v5-EagleX-v2-7B.pth")
# hf_hub_download(repo_id="imthanhlv/binhvq_dedup",local_dir='./',filename="val.jsonl.zst",repo_type="dataset")

###############################
##### 下载repo
###############################
from huggingface_hub import snapshot_download


snapshot_download(repo_id="shibing624/parrots-gpt-sovits-speaker-maimai",repo_type="model",local_dir="model/parrots-gpt-sovits-speaker-maimai")


# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("RWKV/v5-Eagle-7B-HF",trust_remote_code=True)
# tokenizer.save_pretrained('./rwkv_token')

################################
###### 下载魔塔
################################
# from modelscope.models import Model
#
# model = Model.from_pretrained('THUDM/chatglm3-6b-128k',revision="1.14.0")
#
# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b-128k", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/chatglm3-6b-128k", trust_remote_code=True)
# from datasets import load_dataset
#
# data = load_dataset("json","fka/awesome-chatgpt-prompts",split="train")
# print(data)


################################
###### 下载kaggle
################################

import kagglehub

# Download latest version
# import kagglehub
#
# # Download latest version
# path = kagglehub.model_download("google/gemma/tfLite/gemma-1.1-2b-it-cpu-int4")
#
# print("Path to model files:", path)