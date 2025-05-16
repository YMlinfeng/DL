import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CelebADataset(Dataset):

    def __init__(self, root, img_shape=(64, 64)) -> None:
        super().__init__()
        self.root = root
        self.img_shape = img_shape
        self.filenames = sorted(os.listdir(root))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        path = os.path.join(self.root, self.filenames[idx])
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load image {path}. Using a blank image instead.")
            # 返回一张与目标尺寸相同的黑色图片作为替代（注意：这里使用的是 self.img_shape，即训练前置处理目标尺寸）
            img = Image.new('RGB', self.img_shape, (0, 0, 0))
        
        pipeline = transforms.Compose([
            transforms.CenterCrop(168),
            transforms.Resize(self.img_shape),
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))  # 根据需要打开
        ])
        return pipeline(img)


def get_dataloader(root='data/celebA/img_align_celeba', **kwargs):
    dataset = CelebADataset(root, **kwargs)
    return DataLoader(dataset, 16, shuffle=True)


if __name__ == '__main__':
    dataloader = get_dataloader()
    img = next(iter(dataloader))
    print(img.shape)
    # Concat 4x4 images
    N, C, H, W = img.shape
    assert N == 16
    img = torch.permute(img, (1, 0, 2, 3))
    img = torch.reshape(img, (C, 4, 4 * H, W))
    img = torch.permute(img, (0, 2, 1, 3))
    img = torch.reshape(img, (C, 4 * H, 4 * W))
    img = transforms.ToPILImage()(img)
    img.save('work_dirs/tmp.jpg')