import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import pandas as pd
import os
import os
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image
from scipy.linalg import sqrtm
import torch.nn.functional as F
import numpy as np
from scipy.linalg import sqrtm

# 模型路径设置
local_model_path = '/home/lc/Desktop/wza/ly/US-SD/stable-diffusion-v1-4'
device = "cuda:1" if torch.cuda.is_available() else "cpu"

# 加载模型组件
text_encoder = CLIPTextModel.from_pretrained("/sdb4/case/ly/US-SD/stable-diffusion-v1-4/text_encoder")
vae = AutoencoderKL.from_pretrained(
    local_model_path +"/vae",
    local_files_only=True
)
tokenizer = CLIPTokenizer.from_pretrained("/sdb4/case/ly/US-SD/stable-diffusion-v1-4/tokenizer", local_files_only=True)
unet = UNet2DConditionModel.from_pretrained(
    local_model_path + "/unet",
    local_files_only=True
)

vae.to(device)
text_encoder.to(device)
unet.to(device)

def get_text_embeddings(prompt):
    text_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    return text_embeddings

# 定义 14 个类别的 prompts
prompts = [
"Image of Joe Biden",  
"Image of Barack Obama",  
"Image of Hillary Clinton",  
"Image of George Bush",  
"Image of Mike Pence",  
"Image of Ted Cruz",  
"Image of Bernie Sanders",  
"Image of Bill Clinton",  
"Image of Kamala Harris",  
"Image of Alexandria Ocasio-Cortez",  
"Image of Andrew Yang",  
"Image of Ronald Reagan",  
"Image of Ivanka Trump",  
"Image of Melania Trump",  
"Image of Mitch McConnell",  
"Image of Donald Trump",  
]

target_embedding = get_text_embeddings(prompts[-1])

# 计算前 13 个 prompts 与最后一个 prompt 之间的余弦相似度
for i in range(16):
    prompt = prompts[i]
    embedding = get_text_embeddings(prompt)
    target_embedding_flat = target_embedding.view(-1)
    embedding_flat = embedding.view(-1)
    cos_sim = F.cosine_similarity(embedding_flat, target_embedding_flat, dim=0)
    # print(f"Cosine similarity between '{prompt}' and 'Image of english springer': {cos_sim.item()}")
    print(cos_sim.item())

