import torch
import torch.nn.functional as F
from tqdm import tqdm
from model import RefineNet
from utils import compute_metrics


def train_refiner(
    train_loader,
    val_loader,
    epochs=5,
    save_path="refiner_net.pth",
    device=None,
    lr=2e-4
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RefineNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.l1_loss(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[Epoch {epoch+1}] Train Loss: {running_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_psnr, val_ssim, val_lpips = [], [], []
        with torch.no_grad():
            for inputs, targets, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                for i in range(outputs.size(0)):
                    p, s, l = compute_metrics(outputs[i], targets[i])
                    val_psnr.append(p)
                    val_ssim.append(s)
                    val_lpips.append(l)

        print(f"          Val PSNR: {sum(val_psnr)/len(val_psnr):.3f} | SSIM: {sum(val_ssim)/len(val_ssim):.3f} | LPIPS: {sum(val_lpips)/len(val_lpips):.3f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Training complete. Model saved to {save_path}")