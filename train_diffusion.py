import torch
import torch.nn.functional as F
from tqdm import tqdm
from diffusion_model import load_pretrained_diffusion_unet
from utils import compute_metrics
import lpips

def train_diffusion_refiner(
    train_loader,
    val_loader,
    epochs=5,
    save_path="refiner_diffusion_unet.pth",
    device=None,
    lr=2e-4
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pretrained_diffusion_unet().to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(sample=inputs, timestep=torch.tensor([0], device=device)).sample
                # Rescale to [-1, 1] for LPIPS
                outputs_lpips = outputs * 2 - 1
                targets_lpips = targets * 2 - 1
                loss = (
                    0.8 * F.l1_loss(outputs, targets)
                    + 0.2 * lpips_fn(outputs_lpips, targets_lpips).mean()
                )


            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        print(f"[Epoch {epoch+1}] Train Loss: {running_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_psnr, val_ssim, val_lpips = [], [], []
        with torch.no_grad():
            for inputs, targets, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(sample=inputs, timestep=torch.tensor([0], device=device)).sample

                for i in range(outputs.size(0)):
                    p, s, l = compute_metrics(outputs[i], targets[i])
                    val_psnr.append(p)
                    val_ssim.append(s)
                    val_lpips.append(l)

        print(f"Val PSNR: {sum(val_psnr)/len(val_psnr):.3f} | SSIM: {sum(val_ssim)/len(val_ssim):.3f} | LPIPS: {sum(val_lpips)/len(val_lpips):.3f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Training complete. Model saved to {save_path}")
