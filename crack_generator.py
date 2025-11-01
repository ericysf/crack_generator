# !pip install transformers accelerate opencv-python
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
import numpy as np
import torch
import cv2
from PIL import Image
import argparse
import sys

# =========================
# 0️⃣ Parse command line arguments
# =========================
parser = argparse.ArgumentParser(description="ControlNet inpainting with crack generation")
parser.add_argument("image_path", help="Path to the original image")
parser.add_argument("mask_path", help="Path to the mask image")
parser.add_argument("--output_path", default=None, help="Path to save output (default: original_image_with_cracks.png)")
parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
parser.add_argument("--guidance_scale", type=float, default=70, help="Guidance scale for prompt adherence")
parser.add_argument("--controlnet_scale", type=float, default=2.5, help="ControlNet conditioning scale")
parser.add_argument("--inference_steps", type=int, default=200, help="Number of inference steps")

args = parser.parse_args()

# =========================
# 1️⃣ Load images
# =========================
try:
    init_image = load_image(args.image_path)
    mask_image = load_image(args.mask_path)
    print(f"✓ Loaded image from: {args.image_path}")
    print(f"✓ Loaded mask from: {args.mask_path}")
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Resize both to 512x512
# init_image = init_image.resize((512, 512))
# mask_image = mask_image.resize((512, 512))

# =========================
# 2️⃣ Preprocess mask
# =========================
# Convert mask to grayscale numpy array
mask_np = np.array(mask_image.convert("L"))
# Dilate cracks to make them more visible
mask_np = cv2.dilate(mask_np, np.ones((3, 3), np.uint8), iterations=1)
# Optional: convert mask to Canny edges for ControlNet
control_edges = cv2.Canny(mask_np, 100, 200)
control_image = Image.fromarray(control_edges)

# =========================
# 3️⃣ Setup generator
# =========================
generator = torch.Generator(device="cuda").manual_seed(args.seed)

# =========================
# 4️⃣ Load ControlNet + pipeline
# =========================
print("Loading ControlNet model...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
)
print("Loading Stable Diffusion pipeline...")
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
print("✓ Models loaded successfully")

# =========================
# 5️⃣ Define prompt
# =========================
prompt = ("Ultra-realistic high resolution macro photograph of jagged dark deep recessed cracks in a concrete wall, thin hairline fractures blending naturally with the surface, subtle shadow and depth, photorealistic")

#prompt = ("Ultra-realistic macro photograph of deep cracks in surface, dark shadows and black fractures, fine dust debris in crack lines, authentic weathered texture, deep shadows within cracks, photorealistic detail")

#negative_prompt = ("painted, artistic, abstract, unrealistic, blurry, low quality, CGI, rendered, fake, smooth, polished, shallow cracks, light colored cracks")
# =========================
# 6️⃣ Generate image
# =========================
print("Generating image with ControlNet inpainting...")
image = pipe(
    prompt=prompt,
    #negative_prompt=negative_prompt,
    num_inference_steps=args.inference_steps,
    generator=generator,
    eta=1,  
    image=init_image,
    mask_image=mask_image,
    control_image=control_image,
    guidance_scale=args.guidance_scale,
    controlnet_conditioning_scale=args.controlnet_scale
).images[0]

# Save output
if args.output_path is None:
    output_path = args.image_path.replace(".png", "_with_cracks.png")
else:
    output_path = args.output_path

image.save(output_path)
print(f"✓ Image saved to: {output_path}")