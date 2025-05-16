from PIL import Image
import os
from tqdm import tqdm

def validate_images(root):
    files = os.listdir(root)
    for filename in tqdm(files, desc="Validating images"):
        path = os.path.join(root, filename)
        try:
            with Image.open(path) as img:
                img.verify()  # 验证图片是否正常
        except Exception as e:
            print(f"\nRemoving bad image: {path} (error: {e})")
            os.remove(path)

if __name__ == '__main__':
    validate_images('/mnt/bn/occupancy3d/workspace/dataset/img_align_celeba')