import torch.nn as nn
from diffusers import UNet2DModel

def load_pretrained_diffusion_unet(freeze=True):
    model = UNet2DModel.from_pretrained("google/ddpm-celebahq-256")
    if freeze:
        for name, param in model.named_parameters():
            if "up_blocks" not in name and "time_embedding" not in name:
                param.requires_grad = False
    return model