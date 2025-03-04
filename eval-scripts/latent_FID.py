
#                                                    # vae中浅层次特征计算FID
# import argparse
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
# import torchvision.models as models
# import numpy as np
# import os
# import pandas as pd
# from diffusers import (
#     AutoencoderKL,
#     LMSDiscreteScheduler,
#     PNDMScheduler,
#     UNet2DConditionModel,
# )
# from PIL import Image
# from transformers import CLIPTextModel, CLIPTokenizer
# import torch.nn.functional as F
# from scipy.linalg import sqrtm
# from sklearn.decomposition import PCA

# def get_inception_model():
#     model = models.inception_v3(pretrained=True, transform_input=False)
#     model.eval()
#     return model


# def calculate_fid_features(features_original, features_modified, device):
#     """
#     Calculate FID score based on extracted features.

#     Args:
#         features_original (List[torch.Tensor]): List of original features (from image_original).
#         features_modified (List[torch.Tensor]): List of modified features (from image_modified).

#     Returns:
#         float: FID score.
#     """
#     model = get_inception_model().to(device)
#     # Stack the features into a 2D tensor (flattening if necessary)
#     features_original = extract_features(features_original,model, device)
#     features_modified = extract_features(features_modified,model, device)
#     print(f"Original Features: {features_original.shape}")
#     print(f"Modified Features: {features_modified.shape}")

#     # Calculate mean and covariance for original and modified features
#     mu_original = np.mean(features_original, axis=0)
#     mu_modified = np.mean(features_modified, axis=0)

#     cov_original = np.cov(features_original, rowvar=False)
#     cov_modified = np.cov(features_modified, rowvar=False)

#     # Calculate the FID score
#     diff = mu_original - mu_modified
#     cov_sqrt = sqrtm(cov_original.dot(cov_modified))

#     # Handle numerical errors in sqrtm
#     if np.isnan(cov_sqrt).any():
#         cov_sqrt = np.eye(cov_original.shape[0])

#     fid = np.sum(diff**2) + np.trace(cov_original + cov_modified - 2 * cov_sqrt)
#     return fid

# def extract_features(features_input, model, device):
#     print('features_input', len(features_input))
#     features = []
#     with torch.no_grad():
#         print('Type of features_input:', type(features_input))
#         print('Shape of features_input:', [type(f) for f in features_input])
#         features_input = torch.stack(features_input, dim=0)
#         # 如果 features_input 是一个 list，并且里面的元素是数组或张量
#         # 假设 features_input 可能是一个包含张量或多维数组的列表，逐个处理它们
#         if isinstance(features_input, list):
#             features_input = [torch.tensor(f).to(device) if isinstance(f, np.ndarray) else f.to(device) for f in features_input]
#             features_input = torch.stack(features_input)  # 如果 features_input 是多个张量，堆叠成一个张量

#         # 如果 features_input 本身是张量，直接转换
#         elif isinstance(features_input, torch.Tensor):
#             features_input = features_input.squeeze(0).to(device)

#         print('features_input after processing', features_input.shape)
#         feature = model(features_input)
#         print('feature', feature.shape)
#         features = feature.cpu().numpy()
#     return np.array(features)
 

# def generate_images(
#     model_name,
#     prompts_path,
#     save_path,
#     device="cuda:0",
#     guidance_scale=7.5,
#     image_size=512,
#     ddim_steps=100,
#     num_samples=10,
#     from_case=0,
# ):
#     # 1. Load the autoencoder model which will be used to decode the latents into image space.
#     local_model_path = '/home/lc/Desktop/wza/ly/US-SD/stable-diffusion-v1-4'

#     # Load the original (unmodified) model and the modified (forgetting) model
#     text_encoder = CLIPTextModel.from_pretrained("/home/lc/Desktop/wza/ly/US-SD/stable-diffusion-v1-4/text_encoder")

#     vae = AutoencoderKL.from_pretrained(
#         local_model_path + "/vae",
#         local_files_only=True
#     )

#     tokenizer = CLIPTokenizer.from_pretrained("/home/lc/Desktop/wza/ly/US-SD/stable-diffusion-v1-4/tokenizer", local_files_only=True)

#     # Load the original model
#     unet_original = UNet2DConditionModel.from_pretrained(
#         local_model_path + "/unet",
#         local_files_only=True
#     )

#     # Load the forgetting (modified) model
#     unet_modified = UNet2DConditionModel.from_pretrained(
#         local_model_path + "/unet",  # This can point to another folder if you have a separate model for forgetting
#         local_files_only=True
#     )
#     model_path = (
#         f'models/{model_name}/{model_name.replace("compvis","diffusers")}.pt'
#     )
#     # model_path = model_name
#     unet_modified.load_state_dict(torch.load(model_path))
#     # Ensure both models are loaded to the same device
#     scheduler = LMSDiscreteScheduler(
#         beta_start=0.00085,
#         beta_end=0.012,
#         beta_schedule="scaled_linear",
#         num_train_timesteps=1000,
#     )
#     vae.to(device)
#     text_encoder.to(device)
#     unet_original.to(device)
#     unet_modified.to(device)

#     # Load the prompts from CSV
#     df = pd.read_csv(prompts_path)

#     folder_path = f"{save_path}/{model_name}"
#     os.makedirs(folder_path, exist_ok=True)

#     for _, row in df.iterrows():
#         prompt = [str(row.prompt)] * num_samples
#         print(prompt)
#         seed = row.evaluation_seed
#         case_number = row.case_number
#         if case_number < from_case:
#             continue

#         height = image_size
#         width = image_size
#         num_inference_steps = ddim_steps
#         guidance_scale = guidance_scale

#         generator = torch.manual_seed(seed)

#         batch_size = len(prompt)

#         fid_list = []  
#         features_original = [] 
#         features_modified = [] 

#         # Generate images with both models
#         for i in range(1):      ####### num_sample * raund
#             text_input = tokenizer(
#                 prompt,
#                 padding="max_length",
#                 max_length=tokenizer.model_max_length,
#                 truncation=True,
#                 return_tensors="pt",
#             )

#             text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

#             max_length = text_input.input_ids.shape[-1]
#             uncond_input = tokenizer(
#                 [""] * batch_size,
#                 padding="max_length",
#                 max_length=max_length,
#                 return_tensors="pt",
#             )
#             uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

#             text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

#             latents_original = torch.randn(
#                 (batch_size, unet_original.in_channels, height // 8, width // 8),
#                 generator=generator,
#             ).to(device)

#             latents_modified = torch.randn(
#                 (batch_size, unet_modified.in_channels, height // 8, width // 8),
#                 generator=generator,
#             ).to(device)

#             from tqdm.auto import tqdm
#             # Diffusion process for the original model
#             scheduler.set_timesteps(num_inference_steps)
#             latents_original = latents_original * scheduler.init_noise_sigma

#             for t in tqdm(scheduler.timesteps):
#                 latent_model_input = torch.cat([latents_original] * 2)
#                 latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

#                 with torch.no_grad():
#                     noise_pred = unet_original(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

#                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                 noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
#                 latents_original = scheduler.step(noise_pred, t, latents_original).prev_sample

#             # Diffusion process for the modified model
#             scheduler.set_timesteps(num_inference_steps)
#             latents_modified = latents_modified * scheduler.init_noise_sigma


#             for t in tqdm(scheduler.timesteps):
#                 latent_model_input = torch.cat([latents_modified] * 2)
#                 latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

#                 with torch.no_grad():
#                     noise_pred = unet_modified(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

#                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                 noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
#                 latents_modified = scheduler.step(noise_pred, t, latents_modified).prev_sample


#             # Decode the images using VAE
#             latents_original = 1 / 0.18215 * latents_original
#             latents_modified = 1 / 0.18215 * latents_modified

#             with torch.no_grad():
#                 #解码获取图像
#                 shallow_features_original = vae.decode(latents_original).sample
#                 shallow_features_modified = vae.decode(latents_modified).sample
#                  #解码获取图像浅层特征
#                 # shallow_features_original = vae.decode_to_features(latents_original, 'self.conv_act').sample
#                 # shallow_features_modified = vae.decode_to_features(latents_modified, 'self.conv_act').sample

#             # Append extracted features
#             features_original.append(shallow_features_original)
#             features_modified.append(shallow_features_modified)

#             # Calculate FID using features instead of images
#             fid = calculate_fid_features(features_original, features_modified, device)
#             fid_list.append(fid)
#         fid_mean = sum(fid_list) / len(fid_list)

#         print(f"Prompt: {row.prompt} - FID Mean: {fid_mean}")



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         prog="generateImages", description="Generate Images using Diffusers Code"
#     )
#     parser.add_argument("--model_name", help="name of model", type=str, required=True)
#     parser.add_argument(
#         "--prompts_path", help="path to csv file with prompts", type=str, required=True
#     )
#     parser.add_argument(
#         "--save_path", help="folder where to save images", type=str, required=True
#     )
#     parser.add_argument(
#         "--device",
#         help="cuda device to run on",
#         type=str,
#         required=False,
#         default="cuda:0",
#     )
#     parser.add_argument(
#         "--guidance_scale",
#         help="guidance to run eval",
#         type=float,
#         required=False,
#         default=7.5,
#     )
#     parser.add_argument(
#         "--image_size",
#         help="image size used to train",
#         type=int,
#         required=False,
#         default=512,
#     )
#     parser.add_argument(
#         "--from_case",
#         help="continue generating from case_number",
#         type=int,
#         required=False,
#         default=0,
#     )
#     parser.add_argument(
#         "--num_samples",
#         help="number of samples per prompt",
#         type=int,
#         required=False,
#         default=5,
#     )
#     parser.add_argument(
#         "--ddim_steps",
#         help="ddim steps of inference used to train",
#         type=int,
#         required=False,
#         default=100,
#     )
#     args = parser.parse_args()

#     model_name = args.model_name
#     prompts_path = args.prompts_path
#     save_path = args.save_path
#     device = args.device
#     guidance_scale = args.guidance_scale
#     image_size = args.image_size
#     ddim_steps = args.ddim_steps
#     num_samples = args.num_samples
#     from_case = args.from_case

#     generate_images(
#         model_name,
#         prompts_path,
#         save_path,
#         device=device,
#         guidance_scale=guidance_scale,
#         image_size=image_size,
#         ddim_steps=ddim_steps,
#         num_samples=num_samples,
#         from_case=from_case,
#     )















# #                                                  
#                                                           #  latent 的FID计算  (时间过长）)
# import argparse 
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
# import torchvision.models as models
# import numpy as np
# import os
# import pandas as pd
# from diffusers import (
#     AutoencoderKL,
#     LMSDiscreteScheduler,
#     PNDMScheduler,
#     UNet2DConditionModel,
# )
# from PIL import Image
# from transformers import CLIPTextModel, CLIPTokenizer
# import torch.nn.functional as F
# from scipy.linalg import sqrtm


# def calculate_fid(features1, features2):
#     """
#     计算两组特征之间的 FID 值。
#     :param features1: 第一组特征的张量表示
#     :param features2: 第二组特征的张量表示
#     :return: FID 值
#     """
#     mu1 = torch.mean(features1, dim=0)
#     sigma1 = torch.cov(features1.T).cpu() 
#     mu2 = torch.mean(features2, dim=0)
#     sigma2 = torch.cov(features2.T).cpu()
#     diff = mu1 - mu2
#     # covmean = sqrtm(sigma1.cpu() @ sigma2.cpu())
#     covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    
#     # 检查是否为实数
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real
    
#     fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
#     return fid


# def generate_images(
#     model_name,
#     prompts_path,
#     save_path,
#     device="cuda:0",
#     guidance_scale=7.5,
#     image_size=512,
#     ddim_steps=100,
#     num_samples=10,
#     from_case=0,
# ):
#     # 1. Load the autoencoder model which will be used to decode the latents into image space.
#     local_model_path = '/home/lc/Desktop/wza/ly/US-SD/stable-diffusion-v1-4'

#     # Load the original (unmodified) model and the modified (forgetting) model
#     text_encoder = CLIPTextModel.from_pretrained("/home/lc/Desktop/wza/ly/US-SD/stable-diffusion-v1-4/text_encoder")

#     vae = AutoencoderKL.from_pretrained(
#         local_model_path + "/vae",
#         local_files_only=True
#     )

#     tokenizer = CLIPTokenizer.from_pretrained("/home/lc/Desktop/wza/ly/US-SD/stable-diffusion-v1-4/tokenizer", local_files_only=True)

#     # Load the original model
#     unet_original = UNet2DConditionModel.from_pretrained(
#         local_model_path + "/unet",
#         local_files_only=True
#     )

#     # Load the forgetting (modified) model
#     unet_modified = UNet2DConditionModel.from_pretrained(
#         local_model_path + "/unet",  # This can point to another folder if you have a separate model for forgetting
#         local_files_only=True
#     )
#     model_path = (
#         f'models/{model_name}/{model_name.replace("compvis","diffusers")}.pt'
#     )
#     # model_path = model_name
#     unet_modified.load_state_dict(torch.load(model_path))
#     # Ensure both models are loaded to the same device
#     scheduler = LMSDiscreteScheduler(
#         beta_start=0.00085,
#         beta_end=0.012,
#         beta_schedule="scaled_linear",
#         num_train_timesteps=1000,
#     )
#     vae.to(device)
#     text_encoder.to(device)
#     unet_original.to(device)
#     unet_modified.to(device)

#     # Load the prompts from CSV
#     df = pd.read_csv(prompts_path)

#     folder_path = f"{save_path}/{model_name}"
#     os.makedirs(folder_path, exist_ok=True)

#     for _, row in df.iterrows():
#         prompt = [str(row.prompt)] * num_samples
#         print(prompt)
#         seed = row.evaluation_seed
#         case_number = row.case_number
#         if case_number < from_case:
#             continue

#         height = image_size
#         width = image_size
#         num_inference_steps = ddim_steps
#         guidance_scale = guidance_scale

#         generator = torch.manual_seed(seed)

#         batch_size = len(prompt)

#         fid_list = []  

#         # Generate images with both models
#         for i in range(2):      ####### num_sample * raund
#             text_input = tokenizer(
#                 prompt,
#                 padding="max_length",
#                 max_length=tokenizer.model_max_length,
#                 truncation=True,
#                 return_tensors="pt",
#             )

#             text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

#             max_length = text_input.input_ids.shape[-1]
#             uncond_input = tokenizer(
#                 [""] * batch_size,
#                 padding="max_length",
#                 max_length=max_length,
#                 return_tensors="pt",
#             )
#             uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

#             text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

#             latents_original = torch.randn(
#                 (batch_size, unet_original.in_channels, height // 8, width // 8),
#                 generator=generator,
#             ).to(device)

#             latents_modified = torch.randn(
#                 (batch_size, unet_modified.in_channels, height // 8, width // 8),
#                 generator=generator,
#             ).to(device)

#             from tqdm.auto import tqdm
#             # Diffusion process for the original model
#             scheduler.set_timesteps(num_inference_steps)
#             latents_original = latents_original * scheduler.init_noise_sigma

#             for t in tqdm(scheduler.timesteps):
#                 latent_model_input = torch.cat([latents_original] * 2)
#                 latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

#                 with torch.no_grad():
#                     noise_pred = unet_original(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

#                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                 noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
#                 latents_original = scheduler.step(noise_pred, t, latents_original).prev_sample

#             # Diffusion process for the modified model
#             scheduler.set_timesteps(num_inference_steps)
#             latents_modified = latents_modified * scheduler.init_noise_sigma


#             for t in tqdm(scheduler.timesteps):
#                 latent_model_input = torch.cat([latents_modified] * 2)
#                 latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

#                 with torch.no_grad():
#                     noise_pred = unet_modified(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

#                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                 noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
#                 latents_modified = scheduler.step(noise_pred, t, latents_modified).prev_sample


#             # Flatten the latents to calculate FID
#             latents_original_flat = latents_original.view(latents_original.size(0), -1)
#             latents_modified_flat = latents_modified.view(latents_modified.size(0), -1)

#             # Calculate FID in latent space
#             fid = calculate_fid(latents_original_flat, latents_modified_flat)
#             fid_list.append(fid)


#         fid_mean = sum(fid_list) / len(fid_list)


#         print(f"Prompt: {row.prompt} - FID Mean (Latent Space): {fid_mean}")



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         prog="generateImages", description="Generate Images using Diffusers Code"
#     )
#     parser.add_argument("--model_name", help="name of model", type=str, required=True)
#     parser.add_argument(
#         "--prompts_path", help="path to csv file with prompts", type=str, required=True
#     )
#     parser.add_argument(
#         "--save_path", help="folder where to save images", type=str, required=True
#     )
#     parser.add_argument(
#         "--device",
#         help="cuda device to run on",
#         type=str,
#         required=False,
#         default="cuda:0",
#     )
#     parser.add_argument(
#         "--guidance_scale",
#         help="guidance to run eval",
#         type=float,
#         required=False,
#         default=7.5,
#     )
#     parser.add_argument(
#         "--image_size",
#         help="image size used to train",
#         type=int,
#         required=False,
#         default=512,
#     )
#     parser.add_argument(
#         "--from_case",
#         help="continue generating from case_number",
#         type=int,
#         required=False,
#         default=0,
#     )
#     parser.add_argument(
#         "--num_samples",
#         help="number of samples per prompt",
#         type=int,
#         required=False,
#         default=5,
#     )
#     parser.add_argument(
#         "--ddim_steps",
#         help="ddim steps of inference used to train",
#         type=int,
#         required=False,
#         default=100,
#     )
#     args = parser.parse_args()

#     model_name = args.model_name
#     prompts_path = args.prompts_path
#     save_path = args.save_path
#     device = args.device
#     guidance_scale = args.guidance_scale
#     image_size = args.image_size
#     ddim_steps = args.ddim_steps
#     num_samples = args.num_samples
#     from_case = args.from_case

#     generate_images(
#         model_name,
#         prompts_path,
#         save_path,
#         device=device,
#         guidance_scale=guidance_scale,
#         image_size=image_size,
#         ddim_steps=ddim_steps,
#         num_samples=num_samples,
#         from_case=from_case,
#     )


#                                                           images的FID计算
# import argparse
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
# import torchvision.models as models
# import numpy as np
# import os
# import pandas as pd
# from diffusers import (
#     AutoencoderKL,
#     LMSDiscreteScheduler,
#     PNDMScheduler,
#     UNet2DConditionModel,
# )
# from PIL import Image
# from transformers import CLIPTextModel, CLIPTokenizer
# import torch.nn.functional as F
# from scipy.linalg import sqrtm

# def cosine_similarity(emb1, emb2):
#     """
#     计算两个张量之间的余弦相似度
#     :param emb1: 第一个张量，形状为 [1, 77, 768]
#     :param emb2: 第二个张量，形状为 [1, 77, 768]
#     :return: 余弦相似度
#     """
#     # 将张量重塑为 [77 * 768] 并计算余弦相似度
#     cos_sim = F.cosine_similarity(emb1.view(1, -1), emb2.view(1, -1), dim=1)
#     return cos_sim

# def calculate_fid(images1, images2):
#     """
#     计算两组图像之间的 FID 值。
#     :param images1: 第一组图像的张量表示
#     :param images2: 第二组图像的张量表示
#     :return: FID 值
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     inception = models.inception_v3(pretrained=True).to(device).eval()
#     # inception.fc = nn.Identity()  # 移除全连接层，仅使用特征提取部分，使用图像间进行FID计算
#     transform = transforms.Compose([
#         transforms.Resize((512, 512)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
#     ])
#     features1 = []
#     features2 = []
#     for img1, img2 in zip(images1, images2):
#         img1 = transform(img1).unsqueeze(0).to(device)
#         img2 = transform(img2).unsqueeze(0).to(device)
#         with torch.no_grad():
#             feat1 = inception(img1)
#             feat2 = inception(img2)
#         features1.append(feat1.cpu().numpy().squeeze())
#         features2.append(feat2.cpu().numpy().squeeze())
#     features1 = np.array(features1)
#     features2 = np.array(features2)
#     mu1 = np.mean(features1, axis=0)
#     sigma1 = np.cov(features1, rowvar=False)
#     mu2 = np.mean(features2, axis=0)
#     sigma2 = np.cov(features2, rowvar=False)
#     diff = mu1 - mu2
#     covmean = sqrtm(sigma1.dot(sigma2))
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real
#     fid = np.sum(diff ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
#     return fid


# def generate_images(
#     model_name,
#     prompts_path,
#     save_path,
#     device="cuda:0",
#     guidance_scale=7.5,
#     image_size=512,
#     ddim_steps=100,
#     num_samples=10,
#     from_case=0,
# ):
#     # 1. Load the autoencoder model which will be used to decode the latents into image space.
#     local_model_path = '/home/lc/Desktop/wza/ly/US-SD/stable-diffusion-v1-4'

#     # Load the original (unmodified) model and the modified (forgetting) model
#     text_encoder = CLIPTextModel.from_pretrained("/home/lc/Desktop/wza/ly/US-SD/stable-diffusion-v1-4/text_encoder")

#     vae = AutoencoderKL.from_pretrained(
#         local_model_path + "/vae",
#         local_files_only=True
#     )

#     tokenizer = CLIPTokenizer.from_pretrained("/home/lc/Desktop/wza/ly/US-SD/stable-diffusion-v1-4/tokenizer", local_files_only=True)

#     # Load the original model
#     unet_original = UNet2DConditionModel.from_pretrained(
#         local_model_path + "/unet",
#         local_files_only=True
#     )

#     # Load the forgetting (modified) model
#     unet_modified = UNet2DConditionModel.from_pretrained(
#         local_model_path + "/unet",  # This can point to another folder if you have a separate model for forgetting
#         local_files_only=True
#     )
#     model_path = (
#         f'models/{model_name}/{model_name.replace("compvis","diffusers")}.pt'
#     )
#     # model_path = model_name
#     unet_modified.load_state_dict(torch.load(model_path))
#     # Ensure both models are loaded to the same device
#     scheduler = LMSDiscreteScheduler(
#         beta_start=0.00085,
#         beta_end=0.012,
#         beta_schedule="scaled_linear",
#         num_train_timesteps=1000,
#     )
#     vae.to(device)
#     text_encoder.to(device)
#     unet_original.to(device)
#     unet_modified.to(device)

#     # Load the prompts from CSV
#     df = pd.read_csv(prompts_path)

#     folder_path = f"{save_path}/{model_name}"
#     os.makedirs(folder_path, exist_ok=True)

#     # 存储每个 prompt 对应的 text_embeddings
#     prompt_embeddings = {}

#     for _, row in df.iterrows():
#         prompt = [str(row.prompt)] * num_samples
#         print(prompt)
#         seed = row.evaluation_seed
#         case_number = row.case_number
#         if case_number < from_case:
#             continue

#         height = image_size
#         width = image_size
#         num_inference_steps = ddim_steps
#         guidance_scale = guidance_scale

#         generator = torch.manual_seed(seed)

#         batch_size = len(prompt)

#         fid_list = []  

#         # Generate images with both models
#         for i in range(1):      ####### num_sample * raund
#             text_input = tokenizer(
#                 prompt,
#                 padding="max_length",
#                 max_length=tokenizer.model_max_length,
#                 truncation=True,
#                 return_tensors="pt",
#             )

#             text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
#             print('text_embeddings',text_embeddings.shape)
#             prompt_embeddings[row.prompt] = text_embeddings
#             max_length = text_input.input_ids.shape[-1]
#             uncond_input = tokenizer(
#                 [""] * batch_size,
#                 padding="max_length",
#                 max_length=max_length,
#                 return_tensors="pt",
#             )
#             uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

#             text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
#             # print('text_embeddings',text_embeddings.shape)
#             # 存储当前 prompt 的 text_embeddings


#             # latents_original = torch.randn(
#             #     (batch_size, unet_original.in_channels, height // 8, width // 8),
#             #     generator=generator,
#             # ).to(device)

#             # latents_modified = torch.randn(
#             #     (batch_size, unet_modified.in_channels, height // 8, width // 8),
#             #     generator=generator,
#             # ).to(device)

#             # from tqdm.auto import tqdm
#             # # Diffusion process for the original model
#             # scheduler.set_timesteps(num_inference_steps)
#             # latents_original = latents_original * scheduler.init_noise_sigma

#             # for t in tqdm(scheduler.timesteps):
#             #     latent_model_input = torch.cat([latents_original] * 2)
#             #     latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

#             #     with torch.no_grad():
#             #         noise_pred = unet_original(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

#             #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#             #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
#             #     latents_original = scheduler.step(noise_pred, t, latents_original).prev_sample

#             # # Diffusion process for the modified model
#             # scheduler.set_timesteps(num_inference_steps)
#             # latents_modified = latents_modified * scheduler.init_noise_sigma


#             # for t in tqdm(scheduler.timesteps):
#             #     latent_model_input = torch.cat([latents_modified] * 2)
#             #     latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

#             #     with torch.no_grad():
#             #         noise_pred = unet_modified(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

#             #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#             #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
#             #     latents_modified = scheduler.step(noise_pred, t, latents_modified).prev_sample


#             # # Decode the images using VAE
#             # latents_original = 1 / 0.18215 * latents_original
#             # latents_modified = 1 / 0.18215 * latents_modified

#             # with torch.no_grad():
#             #     image_original = vae.decode(latents_original).sample
#             #     image_modified = vae.decode(latents_modified).sample

#             # # Convert to images and save
#             # image_original = (image_original / 2 + 0.5).clamp(0, 1)
#             # image_modified = (image_original / 2 + 0.5).clamp(0, 1)

#             # image_original = image_original.detach().cpu().permute(0, 2, 3, 1).numpy()
#             # image_modified = image_modified.detach().cpu().permute(0, 2, 3, 1).numpy()

#             # images_original = [Image.fromarray((img * 255).round().astype("uint8")) for img in image_original]
#             # images_modified = [Image.fromarray((img * 255).round().astype("uint8")) for img in image_modified]


#             # Calculate FID
#             # fid = calculate_fid(images_original, images_modified)
#             # fid_list.append(fid)


#             # Save the original and modified images
#             # for num, im in enumerate(images_original):
#             #     im.save(f"{folder_path}/{case_number}_{i * 5 + num}_original.png")

#             # for num, im in enumerate(images_modified):
#             #     im.save(f"{folder_path}/{case_number}_{i * 5 + num}_modified.png")


#         # fid_mean = sum(fid_list) / len(fid_list)


#         # print(f"Prompt: {row.prompt} - FID Mean: {fid_mean}")
#     # 计算最后一个 prompt 的 text_embeddings 与其余 prompts 的 text_embeddings 之间的差异度
#     if len(prompt_embeddings) >= 2:
#         last_prompt = list(prompt_embeddings.keys())[-1]
#         last_embedding = prompt_embeddings[last_prompt]
#         differences = []
#         for prompt, embedding in prompt_embeddings.items():
#             if prompt!= last_prompt:
#                 print(prompt)
#                 similarity = cosine_similarity(last_embedding, embedding)
#                 differences.append(similarity)
#             else:
#                 similarity = cosine_similarity(last_embedding, embedding)
#                 differences.append(similarity)
#         print(f"Differences for the last prompt '{last_prompt}': {differences}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         prog="generateImages", description="Generate Images using Diffusers Code"
#     )
#     parser.add_argument("--model_name", help="name of model", type=str, required=True)
#     parser.add_argument(
#         "--prompts_path", help="path to csv file with prompts", type=str, required=True
#     )
#     parser.add_argument(
#         "--save_path", help="folder where to save images", type=str, required=True
#     )
#     parser.add_argument(
#         "--device",
#         help="cuda device to run on",
#         type=str,
#         required=False,
#         default="cuda:0",
#     )
#     parser.add_argument(
#         "--guidance_scale",
#         help="guidance to run eval",
#         type=float,
#         required=False,
#         default=7.5,
#     )
#     parser.add_argument(
#         "--image_size",
#         help="image size used to train",
#         type=int,
#         required=False,
#         default=512,
#     )
#     parser.add_argument(
#         "--from_case",
#         help="continue generating from case_number",
#         type=int,
#         required=False,
#         default=0,
#     )
#     parser.add_argument(
#         "--num_samples",
#         help="number of samples per prompt",
#         type=int,
#         required=False,
#         default=5,
#     )
#     parser.add_argument(
#         "--ddim_steps",
#         help="ddim steps of inference used to train",
#         type=int,
#         required=False,
#         default=100,
#     )
#     args = parser.parse_args()

#     model_name = args.model_name
#     prompts_path = args.prompts_path
#     save_path = args.save_path
#     device = args.device
#     guidance_scale = args.guidance_scale
#     image_size = args.image_size
#     ddim_steps = args.ddim_steps
#     num_samples = args.num_samples
#     from_case = args.from_case

#     generate_images(
#         model_name,
#         prompts_path,
#         save_path,
#         device=device,
#         guidance_scale=guidance_scale,
#         image_size=image_size,
#         ddim_steps=ddim_steps,
#         num_samples=num_samples,
#         from_case=from_case,
#     )


                                                     #latent 欧式距离以及余弦相似度计算
# import argparse
# import os

# import pandas as pd
# import torch
# from diffusers import (
#     AutoencoderKL,
#     LMSDiscreteScheduler,
#     PNDMScheduler,
#     UNet2DConditionModel,
# )
# from PIL import Image
# from transformers import CLIPTextModel, CLIPTokenizer
# import torch.nn.functional as F

# def generate_images(
#     model_name,
#     prompts_path,
#     save_path,
#     device="cuda:0",
#     guidance_scale=7.5,
#     image_size=512,
#     ddim_steps=100,
#     num_samples=10,
#     from_case=0,
# ):
#     # 1. Load the autoencoder model which will be used to decode the latents into image space.
#     local_model_path = '/home/lc/Desktop/wza/ly/US-SD/stable-diffusion-v1-4'

#     # Load the original (unmodified) model and the modified (forgetting) model
#     text_encoder = CLIPTextModel.from_pretrained("/home/lc/Desktop/wza/ly/US-SD/stable-diffusion-v1-4/text_encoder")

#     vae = AutoencoderKL.from_pretrained(
#         local_model_path + "/vae",
#         local_files_only=True
#     )

#     tokenizer = CLIPTokenizer.from_pretrained("/home/lc/Desktop/wza/ly/US-SD/stable-diffusion-v1-4/tokenizer", local_files_only=True)

#     # Load the original model
#     unet_original = UNet2DConditionModel.from_pretrained(
#         local_model_path + "/unet",
#         local_files_only=True
#     )

#     # Load the forgetting (modified) model
#     unet_modified = UNet2DConditionModel.from_pretrained(
#         local_model_path + "/unet",  # This can point to another folder if you have a separate model for forgetting
#         local_files_only=True
#     )
#     model_path = (
#         f'models/{model_name}/{model_name.replace("compvis","diffusers")}.pt'
#     )
#     # model_path = model_name
#     unet_modified.load_state_dict(torch.load(model_path))
#     # Ensure both models are loaded to the same device
#     scheduler = LMSDiscreteScheduler(
#         beta_start=0.00085,
#         beta_end=0.012,
#         beta_schedule="scaled_linear",
#         num_train_timesteps=1000,
#     )
#     vae.to(device)
#     text_encoder.to(device)
#     unet_original.to(device)
#     unet_modified.to(device)

#     # Load the prompts from CSV
#     df = pd.read_csv(prompts_path)

#     folder_path = f"{save_path}/{model_name}"
#     os.makedirs(folder_path, exist_ok=True)

#     for _, row in df.iterrows():
#         prompt = [str(row.prompt)] * num_samples
#         print(prompt)
#         seed = row.evaluation_seed
#         case_number = row.case_number
#         if case_number < from_case:
#             continue

#         height = image_size
#         width = image_size
#         num_inference_steps = ddim_steps
#         guidance_scale = guidance_scale

#         generator = torch.manual_seed(seed)

#         batch_size = len(prompt)

#         euclidean_distance_list = []  #欧氏距离
#         cosine_similarity_list = []  

#         # Generate images with both models
#         for i in range(3):  
#             text_input = tokenizer(
#                 prompt,
#                 padding="max_length",
#                 max_length=tokenizer.model_max_length,
#                 truncation=True,
#                 return_tensors="pt",
#             )

#             text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

#             max_length = text_input.input_ids.shape[-1]
#             uncond_input = tokenizer(
#                 [""] * batch_size,
#                 padding="max_length",
#                 max_length=max_length,
#                 return_tensors="pt",
#             )
#             uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

#             text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

#             latents_original = torch.randn(
#                 (batch_size, unet_original.in_channels, height // 8, width // 8),
#                 generator=generator,
#             ).to(device)

#             latents_modified = torch.randn(
#                 (batch_size, unet_modified.in_channels, height // 8, width // 8),
#                 generator=generator,
#             ).to(device)

#             from tqdm.auto import tqdm
#             # Diffusion process for the original model
#             scheduler.set_timesteps(num_inference_steps)
#             latents_original = latents_original * scheduler.init_noise_sigma

#             for t in tqdm(scheduler.timesteps):
#                 latent_model_input = torch.cat([latents_original] * 2)
#                 latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

#                 with torch.no_grad():
#                     noise_pred = unet_original(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

#                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                 noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
#                 latents_original = scheduler.step(noise_pred, t, latents_original).prev_sample

#             # Diffusion process for the modified model
#             scheduler.set_timesteps(num_inference_steps)
#             latents_modified = latents_modified * scheduler.init_noise_sigma


#             for t in tqdm(scheduler.timesteps):
#                 latent_model_input = torch.cat([latents_modified] * 2)
#                 latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

#                 with torch.no_grad():
#                     noise_pred = unet_modified(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

#                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                 noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
#                 latents_modified = scheduler.step(noise_pred, t, latents_modified).prev_sample

#             latents_difference = latents_original - latents_modified

#             euclidean_distance = torch.norm(latents_difference, p=2, dim=[1, 2, 3]).mean().item()  # 对所有batch计算平均欧氏距离
#             euclidean_distance_list.append(euclidean_distance)

#             latents_original_flat = latents_original.view(batch_size, -1)
#             latents_modified_flat = latents_modified.view(batch_size, -1)
            
#             cosine_sim = F.cosine_similarity(latents_original_flat, latents_modified_flat, dim=1).mean().item()
#             cosine_similarity_list.append(cosine_sim)

#             # Decode the images using VAE
#             latents_original = 1 / 0.18215 * latents_original
#             latents_modified = 1 / 0.18215 * latents_modified

#             with torch.no_grad():
#                 image_original = vae.decode(latents_original).sample
#                 image_modified = vae.decode(latents_modified).sample

#             # Convert to images and save
#             image_original = (image_original / 2 + 0.5).clamp(0, 1)
#             image_modified = (image_modified / 2 + 0.5).clamp(0, 1)

#             image_original = image_original.detach().cpu().permute(0, 2, 3, 1).numpy()
#             image_modified = image_modified.detach().cpu().permute(0, 2, 3, 1).numpy()

#             images_original = (image_original * 255).round().astype("uint8")
#             images_modified = (image_modified * 255).round().astype("uint8")

#             pil_images_original = [Image.fromarray(image) for image in images_original]
#             pil_images_modified = [Image.fromarray(image) for image in images_modified]

#             # Save the original and modified images
#             for num, im in enumerate(pil_images_original):
#                 im.save(f"{folder_path}/{case_number}_{i * 5 + num}_original.png")

#             for num, im in enumerate(pil_images_modified):
#                 im.save(f"{folder_path}/{case_number}_{i * 5 + num}_modified.png")

#         euclidean_distance_mean = sum(euclidean_distance_list) / len(euclidean_distance_list)
#         cosine_similarity_mean = sum(cosine_similarity_list) / len(cosine_similarity_list)

#         print(f"Prompt: {row.prompt} - Euclidean Distance Mean: {euclidean_distance_mean}")
#         print(f"Prompt: {row.prompt} - Cosine Similarity Mean: {cosine_similarity_mean}")



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         prog="generateImages", description="Generate Images using Diffusers Code"
#     )
#     parser.add_argument("--model_name", help="name of model", type=str, required=True)
#     parser.add_argument(
#         "--prompts_path", help="path to csv file with prompts", type=str, required=True
#     )
#     parser.add_argument(
#         "--save_path", help="folder where to save images", type=str, required=True
#     )
#     parser.add_argument(
#         "--device",
#         help="cuda device to run on",
#         type=str,
#         required=False,
#         default="cuda:0",
#     )
#     parser.add_argument(
#         "--guidance_scale",
#         help="guidance to run eval",
#         type=float,
#         required=False,
#         default=7.5,
#     )
#     parser.add_argument(
#         "--image_size",
#         help="image size used to train",
#         type=int,
#         required=False,
#         default=512,
#     )
#     parser.add_argument(
#         "--from_case",
#         help="continue generating from case_number",
#         type=int,
#         required=False,
#         default=0,
#     )
#     parser.add_argument(
#         "--num_samples",
#         help="number of samples per prompt",
#         type=int,
#         required=False,
#         default=5,
#     )
#     parser.add_argument(
#         "--ddim_steps",
#         help="ddim steps of inference used to train",
#         type=int,
#         required=False,
#         default=100,
#     )
#     args = parser.parse_args()

#     model_name = args.model_name
#     prompts_path = args.prompts_path
#     save_path = args.save_path
#     device = args.device
#     guidance_scale = args.guidance_scale
#     image_size = args.image_size
#     ddim_steps = args.ddim_steps
#     num_samples = args.num_samples
#     from_case = args.from_case

#     generate_images(
#         model_name,
#         prompts_path,
#         save_path,
#         device=device,
#         guidance_scale=guidance_scale,
#         image_size=image_size,
#         ddim_steps=ddim_steps,
#         num_samples=num_samples,
#         from_case=from_case,
#     )









                                                 #latent difference image generation
# import argparse
# import os

# import pandas as pd
# import torch
# from diffusers import (
#     AutoencoderKL,
#     LMSDiscreteScheduler,
#     PNDMScheduler,
#     UNet2DConditionModel,
# )
# from PIL import Image
# from transformers import CLIPTextModel, CLIPTokenizer


# def generate_images(
#     model_name,
#     prompts_path,
#     save_path,
#     device="cuda:0",
#     guidance_scale=7.5,
#     image_size=512,
#     ddim_steps=100,
#     num_samples=10,
#     from_case=0,
# ):
#     # 1. Load the autoencoder model which will be used to decode the latents into image space.
#     local_model_path = '/home/lc/Desktop/wza/ly/US-SD/stable-diffusion-v1-4'

#     # Load the original (unmodified) model and the modified (forgetting) model
#     text_encoder = CLIPTextModel.from_pretrained("/home/lc/Desktop/wza/ly/US-SD/stable-diffusion-v1-4/text_encoder")

#     vae = AutoencoderKL.from_pretrained(
#         local_model_path + "/vae",
#         local_files_only=True
#     )

#     tokenizer = CLIPTokenizer.from_pretrained("/home/lc/Desktop/wza/ly/US-SD/stable-diffusion-v1-4/tokenizer", local_files_only=True)

#     # Load the original model
#     unet_original = UNet2DConditionModel.from_pretrained(
#         local_model_path + "/unet",
#         local_files_only=True
#     )

#     # Load the forgetting (modified) model
#     unet_modified = UNet2DConditionModel.from_pretrained(
#         local_model_path + "/unet",  # This can point to another folder if you have a separate model for forgetting
#         local_files_only=True
#     )
#     model_path = (
#         f'models/{model_name}/{model_name.replace("compvis","diffusers")}.pt'
#     )
#     # model_path = model_name
#     unet_modified.load_state_dict(torch.load(model_path))
#     # Ensure both models are loaded to the same device
#     scheduler = LMSDiscreteScheduler(
#         beta_start=0.00085,
#         beta_end=0.012,
#         beta_schedule="scaled_linear",
#         num_train_timesteps=1000,
#     )
#     vae.to(device)
#     text_encoder.to(device)
#     unet_original.to(device)
#     unet_modified.to(device)

#     # Load the prompts from CSV
#     df = pd.read_csv(prompts_path)

#     folder_path = f"{save_path}/{model_name}"
#     os.makedirs(folder_path, exist_ok=True)

#     for _, row in df.iterrows():
#         prompt = [str(row.prompt)] * num_samples
#         print(prompt)
#         seed = row.evaluation_seed
#         case_number = row.case_number
#         if case_number < from_case:
#             continue

#         height = image_size
#         width = image_size
#         num_inference_steps = ddim_steps
#         guidance_scale = guidance_scale

#         generator = torch.manual_seed(seed)

#         batch_size = len(prompt)

#         # Generate images with both models
#         for i in range(3):  # You can adjust the loop as needed
#             text_input = tokenizer(
#                 prompt,
#                 padding="max_length",
#                 max_length=tokenizer.model_max_length,
#                 truncation=True,
#                 return_tensors="pt",
#             )

#             text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

#             max_length = text_input.input_ids.shape[-1]
#             uncond_input = tokenizer(
#                 [""] * batch_size,
#                 padding="max_length",
#                 max_length=max_length,
#                 return_tensors="pt",
#             )
#             uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

#             text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

#             latents_original = torch.randn(
#                 (batch_size, unet_original.in_channels, height // 8, width // 8),
#                 generator=generator,
#             ).to(device)

#             latents_modified = torch.randn(
#                 (batch_size, unet_modified.in_channels, height // 8, width // 8),
#                 generator=generator,
#             ).to(device)
#             from tqdm.auto import tqdm
#             # Diffusion process for the original model
#             scheduler.set_timesteps(num_inference_steps)
#             latents_original = latents_original * scheduler.init_noise_sigma

#             for t in tqdm(scheduler.timesteps):
#                 latent_model_input = torch.cat([latents_original] * 2)
#                 latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

#                 with torch.no_grad():
#                     noise_pred = unet_original(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

#                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                 noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
#                 latents_original = scheduler.step(noise_pred, t, latents_original).prev_sample

#             # Diffusion process for the modified model
#             scheduler.set_timesteps(num_inference_steps)
#             latents_modified = latents_modified * scheduler.init_noise_sigma

#             for t in tqdm(scheduler.timesteps):
#                 latent_model_input = torch.cat([latents_modified] * 2)
#                 latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

#                 with torch.no_grad():
#                     noise_pred = unet_modified(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

#                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                 noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
#                 latents_modified = scheduler.step(noise_pred, t, latents_modified).prev_sample

#             # Calculate the latents difference between the two models
#             latents_difference = torch.abs(latents_original - latents_modified)

#             # Decode the images using VAE
#             latents_original = 1 / 0.18215 * latents_original
#             latents_modified = 1 / 0.18215 * latents_modified
#             latents_difference = 1 / 0.18215 * latents_difference


#             with torch.no_grad():
#                 image_original = vae.decode(latents_original).sample
#                 image_modified = vae.decode(latents_modified).sample
#                 image_difference = vae.decode(latents_difference).sample
#             # Convert to images and save
#             image_original = (image_original / 2 + 0.5).clamp(0, 1)
#             image_modified = (image_modified / 2 + 0.5).clamp(0, 1)
#             image_difference = (image_difference / 2 + 0.5).clamp(0, 1)

#             image_original = image_original.detach().cpu().permute(0, 2, 3, 1).numpy()
#             image_modified = image_modified.detach().cpu().permute(0, 2, 3, 1).numpy()
#             image_difference = image_difference.detach().cpu().permute(0, 2, 3, 1).numpy()

#             images_original = (image_original * 255).round().astype("uint8")
#             images_modified = (image_modified * 255).round().astype("uint8")
#             images_difference = (image_difference * 255).round().astype("uint8")

#             pil_images_original = [Image.fromarray(image) for image in images_original]
#             pil_images_modified = [Image.fromarray(image) for image in images_modified]
#             pil_images_difference = [Image.fromarray(image) for image in images_difference]

#             # Save the original and modified images
#             for num, im in enumerate(pil_images_original):
#                 im.save(f"{folder_path}/{case_number}_{i * 5 + num}_original.png")
            
#             for num, im in enumerate(pil_images_modified):
#                 im.save(f"{folder_path}/{case_number}_{i * 5 + num}_modified.png")

#             for num, im in enumerate(pil_images_difference):
#                 im.save(f"{folder_path}/{case_number}_{i * 5 + num}_difference.png")
#             # Save the latents difference (you can save them as .png or .npy)
#             # latents_diff_image = (latents_difference[0].cpu().numpy() * 255).astype("uint8")
#             # Image.fromarray(latents_diff_image).save(f"{folder_path}/{case_number}_latents_diff_{i}.png")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         prog="generateImages", description="Generate Images using Diffusers Code"
#     )
#     parser.add_argument("--model_name", help="name of model", type=str, required=True)
#     parser.add_argument(
#         "--prompts_path", help="path to csv file with prompts", type=str, required=True
#     )
#     parser.add_argument(
#         "--save_path", help="folder where to save images", type=str, required=True
#     )
#     parser.add_argument(
#         "--device",
#         help="cuda device to run on",
#         type=str,
#         required=False,
#         default="cuda:0",
#     )
#     parser.add_argument(
#         "--guidance_scale",
#         help="guidance to run eval",
#         type=float,
#         required=False,
#         default=7.5,
#     )
#     parser.add_argument(
#         "--image_size",
#         help="image size used to train",
#         type=int,
#         required=False,
#         default=512,
#     )
#     parser.add_argument(
#         "--from_case",
#         help="continue generating from case_number",
#         type=int,
#         required=False,
#         default=0,
#     )
#     parser.add_argument(
#         "--num_samples",
#         help="number of samples per prompt",
#         type=int,
#         required=False,
#         default=5,
#     )
#     parser.add_argument(
#         "--ddim_steps",
#         help="ddim steps of inference used to train",
#         type=int,
#         required=False,
#         default=100,
#     )
#     args = parser.parse_args()

#     model_name = args.model_name
#     prompts_path = args.prompts_path
#     save_path = args.save_path
#     device = args.device
#     guidance_scale = args.guidance_scale
#     image_size = args.image_size
#     ddim_steps = args.ddim_steps
#     num_samples = args.num_samples
#     from_case = args.from_case

#     generate_images(
#         model_name,
#         prompts_path,
#         save_path,
#         device=device,
#         guidance_scale=guidance_scale,
#         image_size=image_size,
#         ddim_steps=ddim_steps,
#         num_samples=num_samples,
#         from_case=from_case,
#     )