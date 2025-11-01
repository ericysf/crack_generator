import cv2
import numpy as np
from PIL import Image
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os
import sys
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='SAM2 Image Segmentation Tool')
parser.add_argument('image_path', type=str, help='Path to the input image')
parser.add_argument('--model', type=str, default='facebook/sam2-hiera-large', 
                    help='SAM2 model to use (default: facebook/sam2-hiera-large)')
args = parser.parse_args()

image_path = args.image_path

# Check if image exists
if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    sys.exit(1)

print("Loading image...")
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)[:, :, ::-1].copy()  # Convert RGB to BGR for OpenCV
folder = os.path.dirname(image_path)
basename = os.path.splitext(os.path.basename(image_path))[0]

# Create output directories if they don't exist
overlay_dir = os.path.join(folder, 'target_overlay')
binary_mask_dir = os.path.join(folder, 'target_binary_mask')

if not os.path.exists(overlay_dir):
    os.makedirs(overlay_dir)
    print(f"Created directory: {overlay_dir}")

if not os.path.exists(binary_mask_dir):
    os.makedirs(binary_mask_dir)
    print(f"Created directory: {binary_mask_dir}")

orig_width, orig_height = image.size
print(f"Original image size: {orig_width}x{orig_height}")

# Resize parameters for display
max_dim = 800
scale = min(max_dim / orig_width, max_dim / orig_height, 1)
display_width = int(orig_width * scale)
display_height = int(orig_height * scale)
display_image_np = cv2.resize(image_np, (display_width, display_height), interpolation=cv2.INTER_AREA)

# Use OpenCV for point selection
clicked_point = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point.clear()
        clicked_point.append((x, y))
        print(f"Clicked point: {clicked_point[0]}")

# Display and get click
window_name = 'Click to select point'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name, display_image_np)
cv2.setMouseCallback(window_name, mouse_callback)
print("Click on the image to select a point, then press any key to continue...")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key != 255:  # Any key pressed
        break
    if clicked_point:  # Point was clicked
        cv2.waitKey(500)  # Wait a bit to see the click
        break

cv2.destroyAllWindows()
cv2.waitKey(1)  # Give time for window to close

if not clicked_point:
    print("No point was clicked. Exiting.")
    sys.exit(1)

# Map clicked coords back to original image coordinates
x_click_resized, y_click_resized = clicked_point[0]
x_orig = int(x_click_resized / scale)
y_orig = int(y_click_resized / scale)
print(f"Mapped clicked point to original image coords: {(x_orig, y_orig)}")

point_coords = np.array([[x_orig, y_orig]])
point_labels = np.array([1])  # foreground label

print(f"Loading SAM2 model: {args.model}...")
try:
    predictor = SAM2ImagePredictor.from_pretrained(args.model)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying with smaller model...")
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")

print("Setting image...")
# Convert back to RGB for SAM2
image_rgb = np.array(image)
predictor.set_image(image_rgb)

print("Running prediction...")
try:
    with torch.inference_mode():
        if torch.cuda.is_available():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                masks, scores, logits = predictor.predict(
                    point_coords=point_coords, 
                    point_labels=point_labels,
                    multimask_output=True
                )
        else:
            masks, scores, logits = predictor.predict(
                point_coords=point_coords, 
                point_labels=point_labels,
                multimask_output=True
            )
except Exception as e:
    print(f"Error during prediction: {e}")
    sys.exit(1)

print(f"Generated {len(masks)} masks with scores: {scores}")

# Use the mask with the highest score
best_mask_idx = np.argmax(scores)
mask_np = masks[best_mask_idx].astype(np.uint8)
print(f"Using mask {best_mask_idx} with score {scores[best_mask_idx]:.3f}")

alpha = 0.5

# Create an RGB red overlay for the mask
red_mask = np.zeros_like(image_np)
red_mask[..., 2] = 255  # Red channel (BGR format)

# Overlay the mask with transparency
overlayed = image_np.copy()
overlayed[mask_np == 1] = (1 - alpha) * overlayed[mask_np == 1] + alpha * red_mask[mask_np == 1]

# Save overlayed image to target_overlay directory
overlayed_rgb = cv2.cvtColor(overlayed.astype(np.uint8), cv2.COLOR_BGR2RGB)
overlayed_img = Image.fromarray(overlayed_rgb)
overlay_path = os.path.join(overlay_dir, f"{basename}_target_overlay_mask.png")
overlayed_img.save(overlay_path)
print(f"Saved overlayed image as '{overlay_path}'")

# Save binary mask to target_binary_mask directory
binary_mask = mask_np * 255  # scale 1 → 255 for white
mask_img = Image.fromarray(binary_mask, mode='L')
output_path = os.path.join(binary_mask_dir, f"{basename}_target_binary_mask.png")
mask_img.save(output_path)
print(f"Saved binary mask as '{output_path}'")

print("\nResults saved successfully!")
print(f"- Overlay: {overlay_path}")
print(f"- Binary mask: {output_path}")

# Optional: Display results (comment out if causing issues)
try:
    display_overlay = cv2.resize(overlayed.astype(np.uint8), (display_width, display_height))
    display_mask = cv2.resize(binary_mask, (display_width, display_height))
    
    cv2.namedWindow('Overlay Result', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Binary Mask', cv2.WINDOW_NORMAL)
    cv2.imshow('Overlay Result', display_overlay)
    cv2.imshow('Binary Mask', display_mask)
    print("\nPress any key to close windows and exit...")
    cv2.waitKey(0)

except:
    print("Could not display results, but files were saved successfully.")
finally:
    cv2.destroyAllWindows()

print("Done!")