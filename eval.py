import os
import argparse

import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import lpips
from skimage.metrics import structural_similarity as compute_ssim

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


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 10 * torch.log10(1 / mse)


def test(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id   

    input_dir = os.path.join(args.base_dir, "Test_Data", args.dataset)
    gt_dir = os.path.join(args.base_dir, "Reference", args.dataset)
    save_dir = os.path.join(args.base_dir, "Results", args.dataset)

    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found -> {input_dir}")
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = ICD(use_gate=args.use_gate).to(device)    
    
    if not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint not found -> {args.ckpt}")
        return

    checkpoint = torch.load(args.ckpt, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
        
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    low_files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(valid_extensions)]

    total_psnr, total_ssim, total_lpips, eval_count = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for img_name in tqdm(low_files, desc="Processing"):
            low_path = os.path.join(input_dir, img_name)
            low_tensor, h_orig, w_orig = load_image(low_path, scale_factor=16)
            
            final_res = model(low_tensor)
            if isinstance(final_res, tuple):
                final_res = final_res[0]
            
            final_res = final_res[:, :, :h_orig, :w_orig]
            final_res = torch.clamp(final_res, 0.0, 1.0)

            save_path = os.path.join(save_dir, img_name)
            torchvision.utils.save_image(final_res, save_path)

            gt_path = os.path.join(gt_dir, img_name)
            
            if os.path.exists(gt_path):
                gt_tensor, _, _ = load_image(gt_path, scale_factor=16)
                gt_tensor = gt_tensor[:, :, :h_orig, :w_orig] 
                
                total_psnr += calculate_psnr(final_res, gt_tensor).item()
                
                img_out_np = final_res.squeeze(0).permute(1, 2, 0).cpu().numpy()
                img_gt_np = gt_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                total_ssim += compute_ssim(img_out_np, img_gt_np, data_range=1.0, channel_axis=-1)
                
                img_res_lpips = final_res * 2.0 - 1.0
                img_gt_lpips = gt_tensor * 2.0 - 1.0
                total_lpips += lpips_fn(img_res_lpips, img_gt_lpips).item()
                    
                eval_count += 1
            else:
                tqdm.write(f"Warning: Reference image not found for {img_name}. Inference only.")

    if eval_count > 0:
        print(f"\nEvaluation Complete (Matched GT pairs: {eval_count})")
        print(f"Average PSNR : {total_psnr / eval_count:.4f} dB")
        print(f"Average SSIM : {total_ssim / eval_count:.4f}")
        print(f"Average LPIPS: {total_lpips / eval_count:.4f}")
    else:
        print("\nInference completed. No reference images found for evaluation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ICD Inference Script")
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--base_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='LOL_real')
    parser.add_argument('--ckpt', type=str, default='./weights/LOL_REAL_best_model.pth')
    parser.set_defaults(use_gate=True)
    
    args = parser.parse_args()
    test(args)