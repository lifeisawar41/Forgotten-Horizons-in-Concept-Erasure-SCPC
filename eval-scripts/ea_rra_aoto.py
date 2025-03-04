import argparse
import os
import sys
import pandas as pd
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from transformers import CLIPTextModel, CLIPTokenizer

def generate_and_calculate(
    model_name,
    prompts_path,
    save_path,
    device="cuda:0",
    guidance_scale=7.5,
    image_size=512,
    ddim_steps=100,
    class_to_forget=0,
    batch_size=5,  # 每次生成的图像数量
):
    # --- 1. 加载模型组件 ---
    local_model_path = '/sdb4/case/ly/US-SD/stable-diffusion-v1-4'
    text_encoder = CLIPTextModel.from_pretrained(f"{local_model_path}/text_encoder")
    vae = AutoencoderKL.from_pretrained(f"{local_model_path}/vae", local_files_only=True)
    tokenizer = CLIPTokenizer.from_pretrained(f"{local_model_path}/tokenizer", local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(f"{local_model_path}/unet", local_files_only=True)
    save_path = f'aoto_EA_RRA_top1/{model_name}.csv'

    # 加载自定义模型权重（如果非SD模型）
    if "SD" not in model_name:
        try:
            model_path = f'models/{model_name}/{model_name.replace("compvis", "diffusers")}.pt'
            unet.load_state_dict(torch.load(model_path))
        except Exception as e:
            print(f"Model path is not valid: {e}")
            sys.exit(1)

    # --- 2. 配置模型和设备 ---
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    # --- 3. 加载分类模型 ---
    classifier = resnet50(weights=ResNet50_Weights.DEFAULT)
    classifier.to(device)
    classifier.eval()
    preprocess = ResNet50_Weights.DEFAULT.transforms()

    # --- 4. 读取prompts文件 ---
    df = pd.read_csv(prompts_path)
    results = {
        "case_number": [],
        "forget_prob": [],
        "retention_prob": []
    }

    # --- 5. 遍历每个prompt生成图像并计算概率 ---
    for _, row in df.iterrows():
        case_number = row.case_number
        prompt = row.prompt
        seed = row.evaluation_seed
        class1 = row.class1

        # 确定生成图像数量：目标类别生成25张，其他类别生成10张
        total_samples = 25 if case_number == class_to_forget else 9
        forget_count = 0
        retention_count = 0

        # 分批生成图像
        for i in range(0, total_samples, batch_size):
            current_batch_size = min(batch_size, total_samples - i)
            prompt_list = [prompt] * current_batch_size

            # --- 5.1 生成图像 ---
            generator = torch.manual_seed(seed)
            text_input = tokenizer(
                prompt_list,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
            uncond_input = tokenizer(
                [""] * current_batch_size,
                padding="max_length",
                max_length=text_input.input_ids.shape[-1],
                return_tensors="pt",
            )
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            latents = torch.randn(
                (current_batch_size, unet.in_channels, image_size//8, image_size//8),
                generator=generator
            ).to(device)
            latents = latents * scheduler.init_noise_sigma

            scheduler.set_timesteps(ddim_steps)
            for t in scheduler.timesteps:
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                with torch.no_grad():
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            latents = 1 / 0.18215 * latents
            with torch.no_grad():
                images = vae.decode(latents).sample
            images = (images / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

            # --- 5.2 分类计算 ---
            for img_array in images:
                img = Image.fromarray((img_array * 255).astype("uint8"))
                batch = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    prediction = classifier(batch).softmax(1)
                # top5_probs, top5_classes = torch.topk(prediction, 5)  # 获取Top5结果  ############## top5
                top5_probs, top5_classes = torch.topk(prediction, 1)  # 获取Top5结果  ############## top1

                # 获取Top5类别名称
                top5_categories = [ResNet50_Weights.DEFAULT.meta["categories"][idx.item()] for idx in top5_classes[0]]
                # print(top5_categories)

                # 检查是否包含目标类别名称
                if case_number == class_to_forget:
                    if class1 not in top5_categories:  # 目标类别名称不在Top5中
                        forget_count += 1
                else:
                    if class1 in top5_categories:  # 其他类别名称在Top5中
                        retention_count += 1

        # --- 5.3 计算概率 ---
        if case_number == class_to_forget:
            forget_prob = forget_count / total_samples
            results["forget_prob"].append(forget_prob)
            results["retention_prob"].append(0.0)  # 目标类别无需保留概率
        else:
            retention_prob = retention_count / total_samples
            results["retention_prob"].append(retention_prob)
            results["forget_prob"].append(0.0)  # 非目标类别无需遗忘概率
        results["case_number"].append(case_number)

    # --- 6. 保存结果 ---
    df_results = pd.DataFrame(results)
    df_merged = pd.merge(df, df_results, on="case_number")
    df_merged.to_csv(save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="GenerateAndCalculate",
        description="Generate images and calculate forget/retention probabilities"
    )
    parser.add_argument("--model_name", help="name of model", type=str, required=True)
    parser.add_argument("--prompts_path", help="path to csv file with prompts", type=str, required=True)
    parser.add_argument("--save_path", help="path to save results", type=str, required=False, default=None)
    parser.add_argument("--device", help="cuda device to run on", type=str, required=False, default="cuda:0")
    parser.add_argument("--guidance_scale", help="guidance to run eval", type=float, required=False, default=7.5)
    parser.add_argument("--image_size", help="image size used to train", type=int, required=False, default=512)
    parser.add_argument("--ddim_steps", help="ddim steps of inference used to train", type=int, required=False, default=100)
    parser.add_argument("--class_to_forget", help="number of samples per prompt", type=int, required=False, default=0)
    parser.add_argument("--batch_size", help="number of images generated per batch", type=int, required=False, default=5)
    args = parser.parse_args()

    generate_and_calculate(
        model_name=args.model_name,
        prompts_path=args.prompts_path,
        save_path=args.save_path,
        device=args.device,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        ddim_steps=args.ddim_steps,
        class_to_forget=args.class_to_forget,
        batch_size=args.batch_size
    )

# import argparse
# import os
# import sys
# import pandas as pd
# import torch
# from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
# from PIL import Image
# from torchvision.models import resnet50, ResNet50_Weights
# from transformers import CLIPTextModel, CLIPTokenizer

# def generate_and_calculate(
#     model_name,
#     prompts_path,
#     save_path,
#     device="cuda:0",
#     guidance_scale=7.5,
#     image_size=512,
#     forget_samples=7,
#     ddim_steps=100,
#     class_to_forget=0,
# ):
#     # --- 1. 加载模型组件 ---
#     local_model_path = '/sdb4/case/ly/US-SD/stable-diffusion-v1-4'
#     text_encoder = CLIPTextModel.from_pretrained(f"{local_model_path}/text_encoder")
#     vae = AutoencoderKL.from_pretrained(f"{local_model_path}/vae", local_files_only=True)
#     tokenizer = CLIPTokenizer.from_pretrained(f"{local_model_path}/tokenizer", local_files_only=True)
#     unet = UNet2DConditionModel.from_pretrained(f"{local_model_path}/unet", local_files_only=True)
#     save_path = f'aoto_EA_RRA/{model_name}.csv'
#     # 加载自定义模型权重（如果非SD模型）
#     if "SD" not in model_name:
#         try:
#             model_path = f'models/{model_name}/{model_name.replace("compvis", "diffusers")}.pt'
#             unet.load_state_dict(torch.load(model_path))
#         except Exception as e:
#             print(f"Model path is not valid: {e}")
#             sys.exit(1)

#     # --- 2. 配置模型和设备 ---
#     vae.to(device)
#     text_encoder.to(device)
#     unet.to(device)
#     scheduler = LMSDiscreteScheduler(
#         beta_start=0.00085,
#         beta_end=0.012,
#         beta_schedule="scaled_linear",
#         num_train_timesteps=1000,
#     )

#     # --- 3. 加载分类模型 ---
#     classifier = resnet50(weights=ResNet50_Weights.DEFAULT)
#     classifier.to(device)
#     classifier.eval()
#     preprocess = ResNet50_Weights.DEFAULT.transforms()

#     # --- 4. 读取prompts文件 ---
#     df = pd.read_csv(prompts_path)
#     results = {
#         "case_number": [],
#         "forget_prob": [],
#         "retention_prob": []
#     }

#     # --- 5. 遍历每个prompt生成图像并计算概率 ---
#     for _, row in df.iterrows():
#         case_number = row.case_number
#         prompt = row.prompt
#         seed = row.evaluation_seed
#         class1 = row.class1
#         if case_number == class_to_forget:
#             num_samples = forget_samples
#         else:
#             num_samples = 5
#         prompt_list = [prompt] * num_samples
#         # --- 5.1 生成图像 ---
#         generator = torch.manual_seed(seed)
#         text_input = tokenizer(
#             prompt_list,
#             padding="max_length",
#             max_length=tokenizer.model_max_length,
#             truncation=True,
#             return_tensors="pt",
#         )
#         text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
#         uncond_input = tokenizer(
#             [""] * num_samples,
#             padding="max_length",
#             max_length=text_input.input_ids.shape[-1],
#             return_tensors="pt",
#         )
#         uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
#         text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

#         latents = torch.randn(
#             (num_samples, unet.in_channels, image_size//8, image_size//8),
#             generator=generator
#         ).to(device)
#         latents = latents * scheduler.init_noise_sigma

#         scheduler.set_timesteps(ddim_steps)
#         for t in scheduler.timesteps:
#             latent_model_input = torch.cat([latents] * 2)
#             latent_model_input = scheduler.scale_model_input(latent_model_input, t)
#             with torch.no_grad():
#                 noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
#             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#             noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
#             latents = scheduler.step(noise_pred, t, latents).prev_sample

#         latents = 1 / 0.18215 * latents
#         with torch.no_grad():
#             images = vae.decode(latents).sample
#         images = (images / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
#         # --- 5.2 分类计算 ---
#         forget_count = 0
#         retention_count = 0

#         for img_array in images:
#             img = Image.fromarray((img_array * 255).astype("uint8"))
#             batch = preprocess(img).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 prediction = classifier(batch).softmax(1)
#             top5_probs, top5_classes = torch.topk(prediction, 5)  # 获取Top5结果
           
#             top5_categories = [ResNet50_Weights.DEFAULT.meta["categories"][idx.item()] for idx in top5_classes[0]]
#             # 检查是否包含目标类别名称

#             if case_number == class_to_forget:
#                 # print(class1)
#                 # print(top5_categories)
#                 if class1 not in top5_categories:  # 目标类别名称不在Top5中
#                     forget_count += 1

#             else:
#                 # print(class1)
#                 # print(top5_categories)
#                 if class1 in top5_categories:  # 其他类别名称在Top5中
#                     retention_count += 1


#         # --- 5.3 计算概率 ---
#         if case_number == class_to_forget:
#             forget_prob = forget_count / num_samples
#             results["forget_prob"].append(forget_prob)
#             results["retention_prob"].append(0.0)  # 目标类别无需保留概率
#         else:
#             retention_prob = retention_count / num_samples
#             results["retention_prob"].append(retention_prob)
#             results["forget_prob"].append(0.0)  # 非目标类别无需遗忘概率
#         results["case_number"].append(case_number)

#     # --- 6. 保存结果 ---
#     df_results = pd.DataFrame(results)
#     df_merged = pd.merge(df, df_results, on="case_number")
#     df_merged.to_csv(save_path, index=False)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         prog="GenerateAndCalculate",
#         description="Generate images and calculate forget/retention probabilities"
#     )
#     parser.add_argument("--model_name", help="name of model", type=str, required=True)
#     parser.add_argument(
#         "--prompts_path", help="path to csv file with prompts", type=str, required=True
#     )
#     parser.add_argument(
#         "--save_path", help="path to save results", type=str, required=False, default=None
#     )
#     parser.add_argument(
#         "--device", help="cuda device to run on", type=str, required=False, default="cuda:0"
#     )
#     parser.add_argument(
#         "--guidance_scale", help="guidance to run eval", type=float, required=False, default=7.5
#     )
#     parser.add_argument(
#         "--image_size", help="image size used to train", type=int, required=False, default=512
#     )
#     # parser.add_argument(
#     #     "--from_case", help="continue generating from case_number", type=int, required=False, default=0
#     # )
#     parser.add_argument(
#         "--forget_samples", help="number of samples per prompt", type=int, required=False, default=5
#     )
#     parser.add_argument(
#         "--ddim_steps", help="ddim steps of inference used to train", type=int, required=False, default=100
#     )
#     parser.add_argument(
#         "--class_to_forget", help="number of samples per prompt", type=int, required=False, default=0
#     )
#     # parser.add_argument("--topk", type=int, required=False, default=5)
#     # parser.add_argument("--batch_size", type=int, required=False, default=250)
#     args = parser.parse_args()

#     generate_and_calculate(
#         model_name=args.model_name,
#         prompts_path=args.prompts_path,
#         save_path=args.save_path,
#         device=args.device,
#         guidance_scale=args.guidance_scale,
#         image_size=args.image_size,
#         forget_samples=args.forget_samples,
#         ddim_steps=args.ddim_steps,
#         class_to_forget=args.class_to_forget
#     )