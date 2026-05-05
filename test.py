import os
import argparse
import time

import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from model import ICD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(img_path, scale_factor=16):
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert('RGB')
    
    img_np = np.asarray(img) / 255.0
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0)
    
    h, w = img_tensor.shape[2], img_tensor.shape[3]
    pad_h = (scale_factor - h % scale_factor) % scale_factor
    pad_w = (scale_factor - w % scale_factor) % scale_factor
    
    if pad_h != 0 or pad_w != 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
    return img_tensor.to(device), h, w


def test(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id   

    input_base = args.input_dir
    save_base = args.save_dir

    if not os.path.exists(input_base):
        print(f"Error: Input directory not found -> {input_base}")
        return

    model = ICD(use_gate=args.use_gate).to(device)    
    
    if not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint not found -> {args.ckpt}")
        return

    checkpoint = torch.load(args.ckpt, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
        
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_paths = []

    for root, dirs, files in os.walk(input_base):
        for f in files:
            if f.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        print(f"Error: No images found in {input_base}")
        return

    print(f"Found {len(image_paths)} images. Starting bulk inference...")

    total_time = 0.0

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Inference"):
            rel_path = os.path.relpath(img_path, input_base)
            save_path = os.path.join(save_base, rel_path)
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            low_tensor, h_orig, w_orig = load_image(img_path, scale_factor=16)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            
            final_res = model(low_tensor)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            if isinstance(final_res, tuple):
                final_res = final_res[0]

            total_time += (end_time - start_time)
            
            final_res = final_res[:, :, :h_orig, :w_orig]
            final_res = torch.clamp(final_res, 0.0, 1.0)

            torchvision.utils.save_image(final_res, save_path)
            
    avg_time = total_time / len(image_paths)
    print(f"\nInference Complete. Processed {len(image_paths)} images.")
    print(f"Average Inference Time: {avg_time:.4f} s ({avg_time*1000:.2f} ms)")
    print(f"Results saved to: {save_base}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ICD Bulk Inference Script")
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--input_dir', type=str, default='./data/Test_Data')
    parser.add_argument('--save_dir', type=str, default='./data/Results')
    parser.add_argument('--disable_gate', action='store_false', dest='use_gate')
    parser.set_defaults(use_gate=True)
    parser.add_argument('--ckpt', type=str, default='./weights/EXPANDLOL_REAL_best_model.pth')
    
    args = parser.parse_args()
    test(args)