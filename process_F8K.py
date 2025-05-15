import os
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Configuration
target_size = (1280, 1280)
image_extensions = {'.jpg', '.jpeg', '.png'}
to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

def pad_to_square_tensor(image_tensor, target_size=(1280, 1280), padding_color=0):
    orig_c, orig_h, orig_w = image_tensor.shape
    scale = min(target_size[0] / orig_w, target_size[1] / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    resized = TF.resize(image_tensor, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)

    pad_h = target_size[1] - new_h
    pad_w = target_size[0] - new_w
    padding = [
        pad_w // 2, pad_h // 2,
        pad_w - pad_w // 2, pad_h - pad_h // 2
    ]
    padded = TF.pad(resized, padding, fill=padding_color)
    return padded

def process_one_image(filename, input_dir, output_dir):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in image_extensions:
        return

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    try:
        image = Image.open(input_path).convert("RGB")
        image_tensor = to_tensor(image)
        padded_tensor = pad_to_square_tensor(image_tensor, target_size)
        padded_image = to_pil(padded_tensor)
        padded_image.save(output_path)
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def process_images_mp(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filenames = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in image_extensions]

    print(f"Using {cpu_count()} CPU cores...")
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(partial(process_one_image, input_dir=input_dir, output_dir=output_dir), filenames),
                  total=len(filenames), desc="Processing images"))

# Entry point
if __name__ == "__main__":
    input_dataset_path = "datasets/Fisheye8K/train/images"
    output_dataset_path = "datasets/Fisheye8K/train/images_padded"

    process_images_mp(input_dataset_path, output_dataset_path)
