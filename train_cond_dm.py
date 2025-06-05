import torch
import torch.nn.functional as F
from tqdm import tqdm
from conditional_diffusion_model import load_conditional_diffusion_unet
from diffusers import DDPMScheduler
from utils import compute_metrics
import lpips

def train_diffusion_refiner(
    train_loader,
    val_loader,
    epochs=5,
    save_path="conditional_refiner_unet.pth",
    device=None,
    lr=2e-4
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_conditional_diffusion_unet(freeze=True).to(device)
    projection = torch.nn.Linear(3, model.config.cross_attention_dim).to(device)

    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, model.parameters())) + list(projection.parameters()),
        lr=lr
    )
    scaler = torch.cuda.amp.GradScaler()
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)

            noise = torch.randn_like(targets)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (inputs.size(0),), device=device).long()
            noisy_targets = scheduler.add_noise(targets, noise, timesteps)
            inputs_ds = F.interpolate(inputs, scale_factor=0.25, mode='bilinear', align_corners=False)
            conditioning = inputs_ds.flatten(2).permute(0, 2, 1)
            conditioning = projection(conditioning)

            with torch.cuda.amp.autocast():
                noise_pred = model(sample=noisy_targets, timestep=timesteps, encoder_hidden_states=conditioning).sample
                loss = F.mse_loss(noise_pred, noise)

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
                noise = torch.randn_like(targets)
                timesteps = torch.tensor([0], device=device).repeat(inputs.size(0))
                noisy_targets = scheduler.add_noise(targets, noise, timesteps)
                conditioning = inputs.flatten(2).permute(0, 2, 1)
                conditioning = projection(conditioning)

                with torch.cuda.amp.autocast():
                    outputs = model(sample=noisy_targets, timestep=timesteps, encoder_hidden_states=conditioning).sample

                for i in range(outputs.size(0)):
                    p, s, l = compute_metrics(outputs[i], targets[i])
                    val_psnr.append(p)
                    val_ssim.append(s)
                    val_lpips.append(l)

        print(f"Val PSNR: {sum(val_psnr)/len(val_psnr):.3f} | SSIM: {sum(val_ssim)/len(val_ssim):.3f} | LPIPS: {sum(val_lpips)/len(val_lpips):.3f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Training complete. Model saved to {save_path}")