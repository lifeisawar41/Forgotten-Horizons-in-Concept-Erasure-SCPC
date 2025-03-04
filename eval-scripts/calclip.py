import os
import clip
import torch
from PIL import Image
import numpy as np

# 定义每个类别和对应的提示（prompt）
category_prompts = {
    '1': "image of trumpet",
    '2': "image of trombone",
    '3': "image of tuba",
    '4': "image of clarinet",
    '5': "image of saxophone",
    '6': "image of flute",
    '7': "image of oboe",
    '8': "image of violin",
    '9': "image of cello",
    '10': "image of double bass",
    '11': "image of piccolo",
    '12': "image of bassoon",
    '13': "image of harp",
    '14': "image of euphonium",
    '15': "image of french horn mouthpiece",
    '16': "image of frenchhorn"

}

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)


def calculate_similarity(image_path, prompt, clip_model, preprocess, device):

    img = Image.open(image_path)
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(img_tensor)

    # 获取文本的嵌入
    text = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)

    similarity = torch.cosine_similarity(image_features, text_features)
    return similarity.item()


def compute_average_similarity(folder_path, category_prompts, clip_model, preprocess, device):
    category_similarities = {category: [] for category in category_prompts}
    for image_name in os.listdir(folder_path):
        if image_name.endswith('.png') or image_name.endswith('.jpg'):
            category_number = image_name.split('_')[0]
            
            prompt = category_prompts.get(category_number, None)
            if prompt is None:
                continue  # 如果没有对应的提示，则跳过

            image_path = os.path.join(folder_path, image_name)
            
            similarity = calculate_similarity(image_path, prompt, clip_model, preprocess, device)
            category_similarities[category_number].append(similarity)
    
    # 计算平均相似度
    average_similarities = {category: np.mean(similarities) if similarities else None
                            for category, similarities in category_similarities.items()}
    return average_similarities

# 使用示例
folder_path = "/sdb4/case/ly/US-SD/final_5_frenchhron_resluts/evalu_5_mymethod_alpha0.5_beta0.01_25epoch/compvis-my-class_5-method_full-alpha_0.5-beta_0.01-epoch_25-lr_1e-05"  # 替换为你的图像文件夹路径
average_similarities = compute_average_similarity(folder_path, category_prompts, clip_model, preprocess, device)

for category, avg_similarity in average_similarities.items():
    print(f"Average similarity for {category}: {avg_similarity}")

