import numpy as np
from PIL import Image
import argparse
import os
import json

def replace_patch_in_image(original_image_path, patch_image_path, coords_json_path, output_path):
    """
    Replace a region in the original image with a patch image using coordinates from JSON.
    
    Args:
        original_image_path: Path to the original image
        patch_image_path: Path to the patch image to insert
        coords_json_path: Path to JSON file containing crop coordinates
        output_path: Path to save the modified image
    
    Returns:
        Modified image with patch replaced
    """
    # Load the original image
    original_image = Image.open(original_image_path)
    
    # Load the patch image
    patch_image = Image.open(patch_image_path)
    
    # Load coordinates from JSON
    with open(coords_json_path, 'r') as f:
        data = json.load(f)
        crop_coords = data['crop_coords']
    
    x_start, y_start, x_end, y_end = crop_coords
    
    # Verify patch dimensions match the coordinate region
    expected_width = x_end - x_start
    expected_height = y_end - y_start
    patch_width, patch_height = patch_image.size
    
    if patch_width != expected_width or patch_height != expected_height:
        print(f"Warning: Patch size ({patch_width}x{patch_height}) doesn't match "
              f"coordinate region ({expected_width}x{expected_height}). Resizing patch...")
        patch_image = patch_image.resize((expected_width, expected_height), Image.LANCZOS)
    
    # Create a copy of the original image to modify
    modified_image = original_image.copy()
    
    # Paste the patch at the specified coordinates
    modified_image.paste(patch_image, (x_start, y_start))
    
    # Save the modified image
    modified_image.save(output_path)
    
    print(f"Modified image saved to {output_path}")
    print(f"Replaced region at coordinates: {crop_coords}")
    
    return modified_image

def main():
    parser = argparse.ArgumentParser(
        description="Replace a patch in the original image using coordinates from JSON."
    )
    parser.add_argument("original", type=str, 
                        help="Path to original image")
    parser.add_argument("patch", type=str, 
                        help="Path to patch image to insert")
    parser.add_argument("coords", type=str, 
                        help="Path to JSON file with crop coordinates")
    parser.add_argument("--output", type=str, default=None, 
                        help="Path to save the modified image (default: same directory as original image with _modified suffix)")
    
    args = parser.parse_args()
    
    # If output not specified, create default path in same directory as original
    if args.output is None:
        original_dir = os.path.dirname(args.original)
        original_basename = os.path.basename(args.original)
        original_name, original_ext = os.path.splitext(original_basename)
        args.output = os.path.join(original_dir, f"{original_name}_modified{original_ext}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    replace_patch_in_image(
        args.original,
        args.patch,
        args.coords,
        args.output
    )

if __name__ == "__main__":
    main()