import numpy as np
from PIL import Image
import argparse
import os
import json

def random_crop_from_mask(image: Image.Image, mask: Image.Image, crop_size=768):
    """
    Randomly crop a patch from the target region defined by the mask.
    If target region smaller than crop_size, crop the region itself.

    Returns:
        cropped_image, cropped_mask, crop_coords (x_start, y_start, x_end, y_end)
    """
    mask_np = np.array(mask)
    
    # Find coordinates of the target region
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No target region found in mask.")

    # Bounding box of target region
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    region_width = x_max - x_min + 1
    region_height = y_max - y_min + 1

    # Determine crop coordinates
    if region_width > crop_size and region_height > crop_size:
        x_start = np.random.randint(x_min, x_max - crop_size + 1)
        y_start = np.random.randint(y_min, y_max - crop_size + 1)
        x_end = x_start + crop_size
        y_end = y_start + crop_size
    else:
        x_start, x_end = x_min, x_max + 1
        y_start, y_end = y_min, y_max + 1

    cropped_image = image.crop((x_start, y_start, x_end, y_end))
    cropped_mask = mask.crop((x_start, y_start, x_end, y_end))

    crop_coords = (x_start, y_start, x_end, y_end)
    return cropped_image, cropped_mask, crop_coords

def main():
    parser = argparse.ArgumentParser(description="Random crop a patch from segmented region.")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("mask", type=str, help="Path to binary mask")
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory (default: current directory)")
    parser.add_argument("--crop_size", type=int, default=768, help="Size of the crop patch (default: 768)")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    image = Image.open(args.image)
    mask = Image.open(args.mask).convert("L")
    
    cropped_img, cropped_mask, crop_coords = random_crop_from_mask(image, mask, args.crop_size)
    
    # Use input image name for outputs
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    
    cropped_img_path = os.path.join(args.out_dir, f"{base_name}_cropped.png")
    cropped_mask_path = os.path.join(args.out_dir, f"{base_name}_cropped_mask.png")
    coords_path = os.path.join(args.out_dir, f"{base_name}_crop_coords.json")
    
    cropped_img.save(cropped_img_path)
    cropped_mask.save(cropped_mask_path)
    
    # Save crop coordinates
    with open(coords_path, "w") as f:
        json.dump({"crop_coords": crop_coords}, f)
    
    print(f"Cropped image saved to {cropped_img_path}")
    print(f"Cropped mask saved to {cropped_mask_path}")
    print(f"Crop coordinates saved to {coords_path}")
    print(f"Crop coordinates: {crop_coords}")

if __name__ == "__main__":
    main()
