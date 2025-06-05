import torch
from diffusers import UNet2DModel, UNet2DConditionModel

def load_conditional_diffusion_unet(freeze=True):
    # Load pretrained UNet2DModel weights
    pretrained = UNet2DModel.from_pretrained("google/ddpm-celebahq-256")

    # Create a new conditional model with matching config (remove invalid encoder_channels)
    config = pretrained.config
    model = UNet2DConditionModel(
        sample_size=config.sample_size,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        layers_per_block=config.layers_per_block,
        block_out_channels=tuple(config.block_out_channels),
        down_block_types=tuple(config.down_block_types),
        up_block_types=tuple(config.up_block_types),
    )

    # Copy weights from pretrained UNet2DModel to UNet2DConditionModel where possible
    pretrained_state = pretrained.state_dict()
    model_state = model.state_dict()

    copied_keys = 0
    for k in model_state:
        if k in pretrained_state and pretrained_state[k].shape == model_state[k].shape:
            model_state[k] = pretrained_state[k]
            copied_keys += 1

    print(f"✅ Copied {copied_keys} pretrained weights into conditional model")
    model.load_state_dict(model_state)

    # Freeze all but up_blocks, time_embedding, and conditioning pathway
    if freeze:
        for name, param in model.named_parameters():
            if "up_blocks" in name or "time_embedding" in name or "encoder" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    return model