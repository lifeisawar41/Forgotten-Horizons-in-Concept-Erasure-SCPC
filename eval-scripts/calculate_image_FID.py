import os
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image
from scipy.linalg import sqrtm
import torch.nn.functional as F

# 计算FID的核心函数
def calculate_fid(real_features, generated_features):

    # 计算均值和协方差矩阵
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_gen = np.mean(generated_features, axis=0)
    sigma_gen = np.cov(generated_features, rowvar=False)

    # 计算FID
    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real.dot(sigma_gen))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum(diff ** 2) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid

# 加载InceptionV3模型
def get_inception_model():
    model = models.inception_v3(pretrained=True, transform_input=False)
    model.eval()
    return model

# 提取图像特征
def extract_features(image_paths, model, device):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    features = []
    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            # print('img', img.shape)
            feature = model(img)
            feature = feature.cpu().numpy()
            feature = feature.squeeze()
            # print('feature', feature.shape)
            if np.isnan(feature).any() or np.isinf(feature).any():
                print(f"Feature for {img_path} contains NaN or inf")
            else:
                features.append(feature)
    return np.array(features)

# 计算每个类别的FID平均值
def compute_category_fid(real_folder, gen_folder, categories, device):
    model = get_inception_model().to(device)
    category_fid = {}

    for category, category_name in categories.items():
        real_images = []
        gen_images = []

        # 根据类别找图像文件
        for filename in os.listdir(real_folder):
            if filename.startswith(str(category)):
                real_images.append(os.path.join(real_folder, filename))

        for filename in os.listdir(gen_folder):
            if filename.startswith(str(category)):
                gen_images.append(os.path.join(gen_folder, filename))

        # 提取特征
        real_features = extract_features(real_images, model, device)
        gen_features = extract_features(gen_images, model, device)
        # print('real_features',real_features.shape)
        # print('gen_features', gen_features.shape)
        # 计算FID
        fid = calculate_fid(real_features, gen_features)
        category_fid[category_name] = fid

    return category_fid

# 主函数
def main():
    real_folder = '/sdb4/case/ly/US-SD/springer_ori_images_fid/compvis-ga'  # 实际图像文件夹路径
    gen_folder = '/sdb4/case/ly/US-SD/final_srpinger_resluts/final_evalu_springer_mymethod_5epoch_alpha0.5_beta0.01_100samples/compvis-my-class_1-method_full-alpha_0.5-beta_0.01-epoch_5-lr_1e-05'  # 生成图像文件夹路径
    categories = {
        1:"Image of table",
        2:"Image of lamp",
        3:"Image of chair",
        4:"Image of Domestic Shorthair Cat",
        5:"Image of Bengal Tiger",
        6:"Image of Amur Leopard",
        7:"Image of Timber Wolf",
        8:"Image of Chihuahua",
        9:"Image of Pug",
        10:"Image of Beagle",
        11:"Image of Springer Spaniel",
        12:"Image of Papillon",
        13:"Image of Cocker Spaniel",
        14:"Image of english springer"
    }

    # 检查CUDA设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    category_fid = compute_category_fid(real_folder, gen_folder, categories, device)

    # 打印FID平均值
    for category_name, fid in category_fid.items():
        print(f"FID for {category_name}: {fid:.4f}")

if __name__ == "__main__":
    main()


