import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torchvision.io import read_image
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from diffusion_model import load_pretrained_diffusion_unet
from utils import compute_metrics
from dataloaders import get_dataloaders


def evaluate_model(
    dataloader: DataLoader,
    weights_path: str,
    device: str = "cuda",
    save_dir: str = None
):
    model = load_pretrained_diffusion_unet(freeze=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    val_psnr, val_ssim, val_lpips = [], [], []
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    with torch.no_grad():
        for i, (inputs, targets, token_ids) in enumerate(tqdm(dataloader, desc="Evaluating")):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(sample=inputs, timestep=torch.tensor([0], device=device)).sample

            for b in range(outputs.size(0)):
                p, s, l = compute_metrics(outputs[b], targets[b])
                val_psnr.append(p)
                val_ssim.append(s)
                val_lpips.append(l)

                if save_dir:
                    token = token_ids[b] if isinstance(token_ids[b], str) else str(token_ids[b])
                    out_dir = os.path.join(save_dir, token)
                    os.makedirs(out_dir, exist_ok=True)
                    save_image(outputs[b].clamp(0, 1), os.path.join(out_dir, "pred.png"))
                    save_image(targets[b].clamp(0, 1), os.path.join(out_dir, "gt.png"))
                    save_image(inputs[b].clamp(0, 1), os.path.join(out_dir, "input.png"))

    print(f"\n✅ Final Evaluation Metrics:")
    print(f"   PSNR: {sum(val_psnr)/len(val_psnr):.3f}")
    print(f"   SSIM: {sum(val_ssim)/len(val_ssim):.3f}")
    print(f"   LPIPS: {sum(val_lpips)/len(val_lpips):.3f}")
    
def show_comparison(token_dir):
    input_img = read_image(os.path.join(token_dir, "input.png")).float() / 255.0
    pred_img = read_image(os.path.join(token_dir, "pred.png")).float() / 255.0
    gt_img = read_image(os.path.join(token_dir, "gt.png")).float() / 255.0

    grid = make_grid([input_img, pred_img, gt_img], nrow=3)
    plt.figure(figsize=(22, 14))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.title("Input | Prediction | Ground Truth")
    plt.show()
    
    print(f"Input shape: {input_img.shape}")
    print(f"Prediction shape: {pred_img.shape}")
    print(f"Ground Truth shape: {gt_img.shape}")