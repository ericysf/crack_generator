import numpy as np
from PIL import Image
import argparse
import os
import json

def replace_patch_in_image(original_image_path, patch_image_path, patch_mask_path, coords_json_path, output_image_path, output_mask_path):
    """
    Replace a region in the original image with a patch image and create a full-size crack mask.
    
    Args:
        original_image_path: Path to the original image
        patch_image_path: Path to the patch image to insert
        patch_mask_path: Path to the patch crack mask
        coords_json_path: Path to JSON file containing crop coordinates
        output_image_path: Path to save the modified image
        output_mask_path: Path to save the full-size crack mask
    
    Returns:
        Tuple of (modified image, full-size crack mask)
    """
    # Load the original image
    original_image = Image.open(original_image_path)
    original_width, original_height = original_image.size
    
    # Load the patch image
    patch_image = Image.open(patch_image_path)
    
    # Load the patch crack mask
    patch_mask = Image.open(patch_mask_path)
    
    # Load coordinates from JSON
    with open(coords_json_path, 'r') as f:
        data = json.load(f)
        crop_coords = data['crop_coords']
    
    x_start, y_start, x_end, y_end = crop_coords
    
    # Verify patch dimensions match the coordinate region
    expected_width = x_end - x_start
    expected_height = y_end - y_start
    patch_width, patch_height = patch_image.size
    patch_mask_width, patch_mask_height = patch_mask.size
    
    if patch_width != expected_width or patch_height != expected_height:
        print(f"Warning: Patch size ({patch_width}x{patch_height}) doesn't match "
              f"coordinate region ({expected_width}x{expected_height}). Resizing patch...")
        patch_image = patch_image.resize((expected_width, expected_height), Image.LANCZOS)
    
    if patch_mask_width != expected_width or patch_mask_height != expected_height:
        print(f"Warning: Patch mask size ({patch_mask_width}x{patch_mask_height}) doesn't match "
              f"coordinate region ({expected_width}x{expected_height}). Resizing patch mask...")
        patch_mask = patch_mask.resize((expected_width, expected_height), Image.LANCZOS)
    
    # Create a copy of the original image to modify
    modified_image = original_image.copy()
    
    # Paste the patch at the specified coordinates
    modified_image.paste(patch_image, (x_start, y_start))
    
    # Create a full-size crack mask (black background)
    # Determine mode based on patch mask (grayscale or binary)
    if patch_mask.mode in ['L', '1']:
        full_mask = Image.new('L', (original_width, original_height), 0)
    elif patch_mask.mode == 'RGB':
        full_mask = Image.new('RGB', (original_width, original_height), (0, 0, 0))
    else:
        # Convert patch mask to grayscale if in unusual mode
        patch_mask = patch_mask.convert('L')
        full_mask = Image.new('L', (original_width, original_height), 0)
    
    # Paste the patch mask at the specified coordinates
    full_mask.paste(patch_mask, (x_start, y_start))
    
    # Save the modified image and full-size mask
    modified_image.save(output_image_path)
    full_mask.save(output_mask_path)
    
    print(f"Modified image saved to {output_image_path}")
    print(f"Full-size crack mask saved to {output_mask_path}")
    print(f"Replaced region at coordinates: {crop_coords}")
    print(f"Original image size: {original_width}x{original_height}")
    print(f"Patch size: {expected_width}x{expected_height}")
    
    return modified_image, full_mask

def main():
    parser = argparse.ArgumentParser(
        description="Replace a patch in the original image and create full-size crack mask using coordinates from JSON."
    )
    parser.add_argument("original", type=str, 
                        help="Path to original image")
    parser.add_argument("patch", type=str, 
                        help="Path to patch image to insert")
    parser.add_argument("patch_mask", type=str, 
                        help="Path to patch crack mask")
    parser.add_argument("coords", type=str, 
                        help="Path to JSON file with crop coordinates")
    parser.add_argument("--output", type=str, default=None, 
                        help="Path to save the modified image (default: same directory as original image with _modified suffix)")
    parser.add_argument("--output_mask", type=str, default=None, 
                        help="Path to save the full-size crack mask (default: same directory as original image with _crack_mask suffix)")
    
    args = parser.parse_args()
    
    # If output not specified, create default path in same directory as original
    if args.output is None:
        original_dir = os.path.dirname(args.original)
        original_basename = os.path.basename(args.original)
        original_name, original_ext = os.path.splitext(original_basename)
        args.output = os.path.join(original_dir, f"{original_name}_modified{original_ext}")
    
    # If output_mask not specified, create default path
    if args.output_mask is None:
        original_dir = os.path.dirname(args.original)
        original_basename = os.path.basename(args.original)
        original_name, original_ext = os.path.splitext(original_basename)
        args.output_mask = os.path.join(original_dir, f"{original_name}_crack_mask{original_ext}")
    
    # Ensure output directories exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    output_mask_dir = os.path.dirname(args.output_mask)
    if output_mask_dir and not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir, exist_ok=True)
    
    replace_patch_in_image(
        args.original,
        args.patch,
        args.patch_mask,
        args.coords,
        args.output,
        args.output_mask
    )

if __name__ == "__main__":
    main()
    
