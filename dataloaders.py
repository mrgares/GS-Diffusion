from torch.utils.data import DataLoader
from image_refinement_dataset import ImageRefinementDataset
from torchvision import transforms

def get_dataloaders(root, batch_size=4, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((256, 416)),
        transforms.ToTensor(),
    ])
    train_ds = ImageRefinementDataset(root, camera_indices=[0,1,2,3,4,5], transform=transform)
    val_ds = ImageRefinementDataset(root, camera_indices=[0], transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
