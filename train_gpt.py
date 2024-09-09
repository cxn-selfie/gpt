# 清除占用显存
#fuser -v /dev/nvidia* 
#kill -9 ***(PID)

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math,inspect

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_embd = config.n_embd
        self.n_head = config.n_head

        # buffer参数类型和parameter参数类型的联系区别 buffer和parameter都有梯度，但优化器只更新parameter参数的梯度
        self.register_buffer("bias", 
                             torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # attn = (q  @  k.transpose(-2,-1)) / k.size(-1)**-0.5
        # # 对attn进行一次mask掩码操作
        # attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # attn = attn.softmax(dim=-1) 
        # y = attn @ v
        # flashAttention 加速训练
        y = F.scaled_dot_product_attention(q, k, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size : int = 1024
    vocab_size : int = 50257
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transfomer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(
                [Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, eps=1e-5)
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #权重共享
        self.lm_head.weight = self.transfomer["wte"].weight

        self.apply(self.__init_weights)

    def __init_weights(self,module):

        std = 0.02
        # 要保持模型参数传递过程中的方差一致性，通过初始化不同的方差值来保证
        # 保持方差一致性可避免梯度爆炸或梯度消失的问题
        if hasattr(module, "NANOGPT_SCALE_INIT"):
            std = (2 ** self.config.n_layer) ** -0.5
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self,weight_decay,learning_rate,device):

        param_dict = { pn:p for pn,p in self.named_parameters() if p.requires_grad }

        decay_params = [p for n,p in param_dict.items() if p.dim()>= 2 ]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2 ]

        optim_groups = [
            {"params" : decay_params, "weight_decay" : weight_decay},
            {"params" : nodecay_params,"weight_decay" : 0.0}
        ]
        
        # import code; code.interact(local=locals())

        num_decay_param =  sum(p.numel() for p in decay_params)
        num_nodecay_param =  sum(p.numel() for p in nodecay_params)

        print(f"num_decay_param : {num_decay_param}, num_nodecay_param : {num_nodecay_param}")


        use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
        print(f"use_fused : {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate,betas = (0.9,0.95),fused = use_fused)
        return optimizer

    def forward(self,idx,target=None):
        B, T = idx.size()
        pos = torch.arange(0,T,dtype=torch.long , device=idx.device)
        tok_emb = self.transfomer["wte"](idx)
        pos_emb = self.transfomer["wpe"](pos)
        x = tok_emb + pos_emb
        for block in self.transfomer["h"]:
            x = block(x)
        x = self.transfomer["ln_f"](x)
        logits = self.lm_head(x)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits,loss
        

    @classmethod
    def from_pretrained(self,model_type):
        assert model_type in ["gpt2", "gpt-menadium","gpt-large", "gpt-xl"]
        config_args = {
            "gpt2" : dict(n_layer=12, n_head=12, n_embd=768),
            "gpt-medium" : dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt-large" : dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt-xl" : dict(n_layer=48, n_head=25, n_embd=1600)
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]
        print("model create end")
        
        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.mask_bias")]

        #对模型初始参数进行赋值
        assert len(sd_hf) == len(sd)
        transpose = ["attn.c_attn.weight","attn.c_proj.weight","mlp.c_fc.weight","mlp.c_proj.weight"]
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transpose):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        with open('/home/lucy/gpt/input.txt', 'r') as f:
            text = f.read()

        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print("tokens len:",len(tokens))
        print("batch len:",len(tokens) // (B * T))
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+ 1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        self.current_position += B*T

        if (self.current_position + B*T +1) >= len(self.tokens):
            self.current_position = 0

        return x,y
        
max_length=30
num_return_sequences=5
max_lr = 6e-5
min_lr = max_lr * 0.1
warm_up = 10
max_step = 50
def get_lr(step):
    if step <= warm_up:
        return max_lr * (step+1) / warm_up
    if step > max_step:
        return min_lr
    decay_ratio = (step - warm_up) / (max_step - warm_up)

    assert decay_ratio >= 0 and decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr * coeff * (max_lr - min_lr)
    


# model = GPT.from_pretrained("gpt2")
# model.eval()
# model.to("cuda")
# sd = model.state_dict()
# print(sd.keys())

model = GPT(GPTConfig())
model.to("cuda")

# 通过减少python开销和减少内存读写次数，加快模型训练速度
model = torch.compile(model)

import tiktoken
import time
enc = tiktoken.get_encoding("gpt2")


torch.cuda.manual_seed(1337)
# torch.cuda.empty_cache()
dataLoader = DataLoaderLite(B=4,T=1024)

#将tensor的数据类型转换成tensorfloat32 
torch.set_float32_matmul_precision("high")

# 判断随机初始化的模型初始化参数是否合理，可通过初始loss进行判断
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

optimizer = model.configure_optimizers(weight_decay=0.01,learning_rate=3e-4,device="cuda")

for step in range(max_step):
    t0 = time.time()
    x,y = dataLoader.next_batch()
    # 在这里加入cuda，能够减少gpu的占用？？
    x,y = x.to("cuda"),y.to("cuda")
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        logits,loss = model(x,y)
        
    # 用这种方式打断点
    # import code; code.interact(local=locals())
    loss.backward()
    # 梯度裁减，防止梯度爆炸
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    ds = (t1-t0) * 1000
    print({"step":step,"lr":lr,"loss:":loss.item(),"time(ms):":ds})

# print(loss)


# tokens = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens,dtype=torch.long).unsqueeze(0).repeat(num_return_sequences,1)
# x = tokens.to("cuda")

# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     logits = model(x)
#     logits = logits[:, -1, :]
#     logits = logits.softmax(dim=-1)
#     # 只从前50个token中采样，避免模型采样到非常离谱的token,迷失方向
#     topk_probs,topk_indices = torch.topk(logits, 50, dim=-1)
#     ix = torch.multinomial(topk_probs, num_samples=1)
#     xcol = torch.gather(topk_indices,-1, ix)
#     x = torch.cat((x, xcol), dim=1)

# for i in range(num_return_sequences):
#     decode = enc.decode(x[i,0:max_length].tolist()) 
#     print(decode)
