import argparse
import os
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import torch
from convertModels import savemodelDiffusers
from dataset import setup_forget_data, setup_model, setup_remain_data
from diffusers import LMSDiscreteScheduler
from tqdm import tqdm

#球体采样法
def perturb_condition(c, epsilon=0.001):
        """
        在球体范围内初始化扰动
        :param c: 原始点
        :param epsilon: 球体半径
        :return: 在球体范围内采样的扰动条件 c'
        """
        c = c.float()
        
        # 直接在球体范围内采样初始化扰动
        perturbation = torch.randn_like(c)  # 正态分布
        perturbation = perturbation / perturbation.norm(p=2, dim=-1, keepdim=True)  # 单位化
        # print(perturbation.shape)
        perturbation = perturbation * torch.rand(c.shape[0], c.shape[1], 1, device=c.device).sqrt() * epsilon  # 缩放到球体内

        # 初始化偏移量 u
        u = torch.zeros_like(c, requires_grad=True)  

        # 计算 c'
        c_prime = c + perturbation + u

        return c_prime

def gradient_ascent(
    class_to_forget,
    train_method,
    alpha,
    beta,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    mask_path,
    diffusers_config_path,
    device,
    image_size=512,
    ddim_steps=50,
):
    # MODEL TRAINING SETUP
    # model = setup_model(config_path, ckpt_path, device)
    criteria = torch.nn.MSELoss()
    model_tea = setup_model(config_path, ckpt_path, device)
    model_stu = setup_model(config_path, ckpt_path, device)
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    remain_dl, descriptions = setup_remain_data(class_to_forget, batch_size, image_size,max_samples_per_class=10)
    forget_dl, _ = setup_forget_data(class_to_forget, batch_size, image_size,max_samples_per_class=5)
    class_to_posedo = (int(class_to_forget) + 1) % 10
    print(class_to_posedo)
    posedo_dl, _ = setup_forget_data(class_to_posedo, batch_size, image_size,max_samples_per_class=10)

    # set model to train
    model_stu.train()
    model_tea.eval()
    losses = []

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model_stu.model.diffusion_model.named_parameters():
        # train only x attention layers
        if train_method == "xattn":
            if "attn2" in name:
                parameters.append(param)
        # train all layers
        if train_method == "full":
            parameters.append(param)

    optimizer = torch.optim.Adam(parameters, lr=lr)

    if mask_path:
        mask = torch.load(mask_path)
        name = f"compvis-my-mask-method_{train_method}-alpha_{alpha}-beta_{beta}-epoch_{epochs}-lr_{lr}"
    else:
        name = f"compvis-my-class_{str(class_to_forget)}-method_{train_method}-alpha_{alpha}-beta_{beta}-epoch_{epochs}-lr_{lr}"


    # TRAINING CODE
    for epoch in range(epochs):
        with tqdm(total=len(posedo_dl)) as time:
            for i in range(len(posedo_dl)):
                optimizer.zero_grad()

                forget_images, forget_labels = next(iter(forget_dl))
                remain_images, remain_labels = next(iter(remain_dl))
                posedo_images, posedo_labels = next(iter(posedo_dl))


                pseudo_prompts = [
                    descriptions[(int(class_to_forget) + 1) % 10]
                    for label in forget_labels
                ]

                forget_prompts = [descriptions[label] for label in forget_labels]
                remain_prompts = [descriptions[label] for label in remain_labels]

                forget_batch = {
                    "jpg": posedo_images.permute(0, 2, 3, 1),
                    "txt": forget_prompts,
                }

                forget_loss = model_stu.shared_step(forget_batch)[0]

                remain_batch = {
                    "jpg": remain_images.permute(0, 2, 3, 1),
                    "txt": remain_prompts,
                }
                remain_loss = model_stu.shared_step(remain_batch)[0]


                #获取大的emb空间进行邻近采样
                forget_input, forget_emb = model_stu.get_input(
                    forget_batch, model_stu.first_stage_key
                )
                t = torch.randint(
                    800,
                    model_stu.num_timesteps,
                    (forget_input.shape[0],),
                    device=model_stu.device,
                ).long()
                noise = torch.randn_like(forget_input, device=model_stu.device)
                # total_emb = torch.cat((remain_emb, forget_emb), dim=0)
                # similar_emb = generate_similar_embedding_linear(remain_emb, forget_emb, similarity_level=0.99).to(model_stu.device)
                similar_emb =  perturb_condition(forget_emb, epsilon=0.001)
                similar_loss =  torch.nn.functional.mse_loss(similar_emb, forget_emb)   #embedding的loss
                similar_input = torch.randn_like(forget_input, device=model_stu.device)
                similar_noisy = model_stu.q_sample(x_start=similar_input, t=t, noise=noise)
                similar_out_stu = model_stu.apply_model(similar_noisy, t, similar_emb)
                similar_out_tea = model_tea.apply_model(similar_noisy, t, similar_emb) #eval
                teach_loss = criteria(similar_out_stu, similar_out_tea)


                loss = forget_loss + alpha * remain_loss + beta * teach_loss + beta * similar_loss
                loss.backward()
                losses.append(loss.item() / batch_size)

                if mask_path:
                    for n, p in model.named_parameters():
                        if p.grad is not None:
                            p.grad *= mask[n.split("model.diffusion_model.")[-1]].to(
                                device
                            )

                optimizer.step()
                time.set_description("Epoch %i" % epoch)
                time.set_postfix(loss=loss.item() / batch_size)
                sleep(0.1)
                time.update(1)
        print('forget_loss' , forget_loss)
        print('teach_loss' , teach_loss)
        print('similar_loss' , similar_loss)

    model_stu.eval()
    save_model(
        model_stu,
        name,
        4,
        save_compvis=True,
        save_diffusers=True,
        compvis_config_file=config_path,
        diffusers_config_file=diffusers_config_path,
    )
    save_history(losses, name, classes)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_loss(losses, path, word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f"{word}_loss")
    plt.legend(loc="upper left")
    plt.title("Average loss in trainings", fontsize=20)
    plt.xlabel("Data point", fontsize=16)
    plt.ylabel("Loss value", fontsize=16)
    plt.savefig(path)


def save_model(
    model,
    name,
    num,
    compvis_config_file=None,
    diffusers_config_file=None,
    device="cpu",
    save_compvis=True,
    save_diffusers=True,
):
    # SAVE MODEL
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f"{folder_path}/{name}-epoch_{num}.pt"
    else:
        path = f"{folder_path}/{name}.pt"
    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print("Saving Model in Diffusers Format")
        savemodelDiffusers(
            name, compvis_config_file, diffusers_config_file, device=device
        )


def save_history(losses, name, word_print):
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses, f"{folder_path}/loss.png", word_print, n=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train", description="train a stable diffusion model from scratch"
    )
    parser.add_argument(
        "--class_to_forget",
        help="class corresponding to concept to erase",
        type=str,
        required=True,
        default="0",
    )
    parser.add_argument(
        "--train_method", help="method of training", type=str, required=True
    )
    parser.add_argument(
        "--alpha",
        help="guidance of start image used to train",
        type=float,
        required=False,
        default=0.1,
    )
    parser.add_argument(
        "--batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=2,
    )
    parser.add_argument(
        "--epochs", help="epochs used to train", type=int, required=False, default=10
    )
    parser.add_argument(
        "--lr",
        help="learning rate used to train",
        type=float,
        required=False,
        default=1e-5,
    )
    parser.add_argument(
        "--ckpt_path",
        help="ckpt path for stable diffusion v1-4",
        type=str,
        required=False,
        default="/home/lc/Desktop/wza/ly/selective-amnesia/sd/sd-v1-4-full-ema.ckpt",
    )
    parser.add_argument(
        "--mask_path",
        help="mask path for stable diffusion v1-4",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--beta",
        help="guidance of start image used to train",
        type=float,
        required=False,
        default=0.1,
    )
    parser.add_argument(
        "--config_path",
        help="config path for stable diffusion v1-4 inference",
        type=str,
        required=False,
        default="configs/stable-diffusion/v1-inference.yaml",
    )
    parser.add_argument(
        "--diffusers_config_path",
        help="diffusers unet config json path",
        type=str,
        required=False,
        default="diffusers_unet_config.json",
    )
    parser.add_argument(
        "--device",
        help="cuda devices to train on",
        type=str,
        required=False,
        default="4",
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--ddim_steps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=50,
    )
    args = parser.parse_args()

    classes = int(args.class_to_forget)
    train_method = args.train_method
    alpha = args.alpha
    beta = args.beta
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    ckpt_path = args.ckpt_path
    mask_path = args.mask_path
    config_path = args.config_path
    diffusers_config_path = args.diffusers_config_path
    device = f"cuda:{int(args.device)}"
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    gradient_ascent(
        classes,
        train_method,
        alpha,
        beta,
        batch_size,
        epochs,
        lr,
        config_path,
        ckpt_path,
        mask_path,
        diffusers_config_path,
        device,
        image_size,
        ddim_steps,
    )