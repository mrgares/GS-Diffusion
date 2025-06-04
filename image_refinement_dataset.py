import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageRefinementDataset(Dataset):
    def __init__(self, root_dir, camera_indices=[1, 2, 3, 4, 5], transform=None):
        self.root_dir = root_dir
        self.camera_indices = camera_indices
        self.transform = transform or transforms.ToTensor()
        self.samples = []

        for token in os.listdir(root_dir):
            token_path = os.path.join(root_dir, token)
            if not os.path.isdir(token_path):
                continue
            for cam_idx in self.camera_indices:
                pred_img = os.path.join(token_path, f"{cam_idx}.png")
                gt_img = os.path.join(token_path, f"{cam_idx}_gt.png")
                if os.path.exists(pred_img) and os.path.exists(gt_img):
                    self.samples.append((pred_img, gt_img, token))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_path, target_path, token = self.samples[idx]
        input_img = Image.open(input_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")
        return self.transform(input_img), self.transform(target_img), token
