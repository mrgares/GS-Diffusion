import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lpips_fn = lpips.LPIPS(net='alex').to(device)

@torch.no_grad()
def compute_metrics(pred, gt):
    pred = pred.to(device)
    gt = gt.to(device)

    pred_np = pred.permute(1, 2, 0).cpu().numpy()
    gt_np = gt.permute(1, 2, 0).cpu().numpy()

    psnr_val = psnr(gt_np, pred_np, data_range=1.0)
    ssim_val = ssim(gt_np, pred_np, channel_axis=-1, data_range=1.0)

    lpips_val = lpips_fn(pred.unsqueeze(0), gt.unsqueeze(0)).item()

    return psnr_val, ssim_val, lpips_val