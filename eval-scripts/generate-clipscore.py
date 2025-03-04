import os
import clip
import torch
import argparse
import pandas as pd
from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from PIL import Image
import numpy as np
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from types import MethodType
from typing import Union
import numpy as np
import sys

def read_prompts_from_csv(prompts_path):
    df = pd.read_csv(prompts_path)
    prompts = []
    seeds = []
    case_numbers = []
    for _, row in df.iterrows():
        prompts.append(str(row.prompt))
        seeds.append(row.evaluation_seed)
        case_numbers.append(row.case_number)
    return prompts, seeds, case_numbers


def calculate_similarity(image, prompt, clip_model, preprocess, device):
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(img_tensor)
    text = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
    similarity = torch.cosine_similarity(image_features, text_features)
    return similarity.item()

# 生成图像并计算 CLIP Score
def generate_and_compute_clipscore(model_name, prompts_path, device="cuda:0", guidance_scale=7.5, image_size=512,
                                   ddim_steps=100, num_samples=5):

    local_model_path = '/sdb4/case/ly/US-SD/stable-diffusion-v1-4'
    vae = AutoencoderKL.from_pretrained(
        local_model_path + "/vae",
        local_files_only=True
    )
    text_encoder = CLIPTextModel.from_pretrained("/sdb4/case/ly/US-SD/stable-diffusion-v1-4/text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained("/sdb4/case/ly/US-SD/stable-diffusion-v1-4/tokenizer", local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(
        local_model_path + "/unet",
        local_files_only=True
    )
    if "SD" not in model_name:
        try:
            model_path = (
                f'models/{model_name}/{model_name.replace("compvis", "diffusers")}.pt'
            )
            unet.load_state_dict(torch.load(model_path))
        except Exception as e:
            print(
                f"Model path is not valid, please check the file name and structure: {e}"
            )
            # sys.exit(1)
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)


    prompts, seeds, case_numbers = read_prompts_from_csv(prompts_path)

    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    category_clipscores = {}

    for prompt, seed, case_number in zip(prompts, seeds, case_numbers):
        prompt_list = [prompt] * num_samples
        print(prompt_list)
        generator = torch.manual_seed(seed)
        batch_size = len(prompt_list)

        text_input = tokenizer(
            prompt_list,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, unet.in_channels, image_size // 8, image_size // 8),
            generator=generator,
        )
        latents = latents.to(device)

        scheduler.set_timesteps(ddim_steps)
        latents = latents * scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(
                latent_model_input, timestep=t
            )
            with torch.no_grad():
                noise_pred = unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
            )
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(img) for img in images]

        if case_number not in category_clipscores:
            category_clipscores[case_number] = []

        for im in pil_images:
            clipscore = calculate_similarity(im, prompt, clip_model, preprocess, device)
            category_clipscores[case_number].append(clipscore)

    # 计算每个类别的平均 CLIP Score
    category_avg_clipscores = {
        category: np.mean(scores) if scores else None
        for category, scores in category_clipscores.items()
    }

    return category_avg_clipscores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generateAndComputeClipscore", description="Generate Images and Compute CLIP Score"
    )
    parser.add_argument("--train_method", help="training method", type=str, default='full')
    parser.add_argument("--alpha", help="alpha value", type=float, default=0.5)
    parser.add_argument("--beta", help="beta value", type=float, default=0.01)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-05)
    parser.add_argument("--epochs", help="number of epochs", type=int, default=10)
    parser.add_argument("--class_to_forget", help="class to forget", type=int, required=True)
    parser.add_argument("--device", help="cuda device to run on", type=str, default="cuda:0")
    parser.add_argument("--guidance_scale", help="guidance to run eval", type=float, default=7.5)
    parser.add_argument("--image_size", help="image size used to train", type=int, default=512)
    parser.add_argument("--ddim_steps", help="ddim steps of inference used to train", type=int, default=50)
    parser.add_argument("--num_samples", help="number of samples per prompt", type=int, default=5)
    parser.add_argument("--method", help="number of samples per prompt", type=str, default='my')
    args = parser.parse_args()

    class_to_forget = args.class_to_forget
    prompts_path = f"prompts/class{class_to_forget}_prompts_new.csv"
    # prompts_path = 'prompts/classnude_prompts_new.csv'    ############## nude
    # 动态生成 model_name
# 如果选择的方法是 'my'
    if args.method == 'my':
        model_name = f"compvis-my-class_{str(class_to_forget)}-method_{args.train_method}-alpha_{args.alpha}-beta_{args.beta}-epoch_{args.epochs}-lr_{args.lr}"

    # 如果选择的方法是 'ga'
    if args.method == 'ga':
        model_name = f"compvis-ga-method_{args.train_method}-class_{str(class_to_forget)}-alpha_{args.alpha}-epoch_{args.epochs}-lr_{args.lr}"

    # 如果选择的方法是 'esd'
    if args.method == 'esd':
        model_name = f"compvis-esd-class_{str(class_to_forget)}-method_{args.train_method}-lr_{args.lr}-iterations_{args.epochs}"
        # model_name = f"compvis-esd-class_{str(class_to_forget)}-method_{args.train_method}-lr_{args.lr}"  #eval
    # 如果选择的方法是 'salun'
    if args.method == 'salun':
        model_name = f"compvis-cl-mask-class_{str(class_to_forget)}-method_{args.train_method}-alpha_{args.alpha}-epoch_{args.epochs}-lr_{args.lr}"
    # 如果选择的方法是 'ori'   
    if args.method == 'ori':
        model_name = f"ori_SD1.4_{str(class_to_forget)}"

    if args.method == 'my3':
        model_name = f"compvis-my-class_1-np_3-method_xattn-alpha_0.5-beta_0.01-epoch_30-lr_1e-05"
    if args.method == 'my5':
        model_name = f"compvis-my-class_1-np_5-method_xattn-alpha_0.5-beta_0.01-epoch_30-lr_1e-05"
    if args.method == 'my7':
        model_name = f"compvis-my-class_1-np_7-method_xattn-alpha_0.5-beta_0.01-epoch_30-lr_1e-05"
    category_avg_clipscores = generate_and_compute_clipscore(
        model_name,
        prompts_path,
        device=args.device,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        ddim_steps=args.ddim_steps,
        num_samples=args.num_samples
    )

    # 输出每个类别的平均 CLIP Score
    for category, avg_clipscore in category_avg_clipscores.items():
        print(f"Average CLIP Score for category {category}: {avg_clipscore}")
    # 生成 CSV 文件名
    if args.method == 'ga':
        csv_filename = f"auto_results_ga/{class_to_forget}_ga_results.csv"

    elif args.method == 'my3':
        csv_filename = f"auto_results_my/{class_to_forget}_auto_results_np3_xattn.csv"
        # csv_filename = f"auto_results_my/nude_auto_results.csv"   #####
    elif args.method == 'my5':
        csv_filename = f"auto_results_my/{class_to_forget}_auto_results_np5_xattn.csv"

    elif args.method == 'my7':
        csv_filename = f"auto_results_my/{class_to_forget}_auto_results_np7_xattn.csv"
    elif args.method == 'esd':
        csv_filename = f"auto_results_esd/{class_to_forget}_esd_results.csv"

    elif args.method == 'salun':
        csv_filename = f"auto_results_salun/{class_to_forget}_salun_results.csv"

    elif args.method == 'ori':
        # csv_filename = f"aoto_results_ori/{class_to_forget}_ori_results.csv"
        csv_filename = f"auto_results_my/nude_auto_results.csv" 
    else:
        raise ValueError(f"Unsupported method: {args.method}")
    # 检查文件是否存在 
    file_exists = os.path.isfile(csv_filename)

    # 读取现有的 CSV 文件（如果存在）
    if file_exists:
        existing_df = pd.read_csv(csv_filename, index_col=0)
    else:
        existing_df = pd.DataFrame()

    # 准备新的数据
    new_data = {args.epochs: category_avg_clipscores}
    new_df = pd.DataFrame(new_data) # 转置，使 epoch 为行，类别为列

    # 合并数据
    combined_df = pd.concat([existing_df, new_df], axis=0)

    # 写入 CSV 文件
    combined_df.to_csv(csv_filename, sep=' ', na_rep='nan')

    # 输出最后一个类别的平均 CLIP Score 和所有类别的排序结果
    scores = list(category_avg_clipscores.values())
    scores.sort()
    last_category_score = list(category_avg_clipscores.values())[-1]
    print(f"Last category score: {last_category_score}")
    print(f"Sorted scores: {scores}")