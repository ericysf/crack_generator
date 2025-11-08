

# Crack Generator
A synthetic crack generation system for building inspection and damage assessment applications. This tool uses SAM2 object segmentation and ControlNet-guided Stable Diffusion to generate realistic cracks on building walls.  
![Pipeline Overview](Pipeline.png)


## Overview

The pipeline consists of several stages:  

1. **Target Region Identification**: SAM2 segments the region where cracks should appear  
2. **Patch Extraction**: Crops a 768×768 patch from the target region  
3. **Crack Mask Generation**: Creates realistic binary crack patterns  
4. **Synthetic Crack Generation**: Uses ControlNet to guide Stable Diffusion for photorealistic crack synthesis  
5. **Full-Size Reconstruction**: Recreates full-resolution images with generated cracks  

## Installation

### Clone the Repository

```bash  
git clone https://github.com/ericysf/crack_generator.git  
cd crack_generator  
```  

### Prerequisites  

```bash  
pip install -r requirements.txt  
```  

### SAM2 Setup

```bash  
# Clone SAM2 repository  
git clone https://github.com/facebookresearch/sam2.git  
cd sam2  
  
# Install SAM2  
pip install -e .  
  
# Download checkpoints  
cd checkpoints  
./download_ckpts.sh  
cd ../..  
  
# Copy segmentation script to SAM2 folder  
cp sam2_segmentation.py sam2/sam2_segmentation.py  
```   
  
## Usage
  
### 1. Target Region Segmentation  
  
Identify the target region for crack generation using interactive segmentation.  
  
**Required Arguments:**  
- `image_path`: Path to the image to be segmented  
  
**Optional Arguments:**  
- `--model`: SAM2 model to use (default: `facebook/sam2-hiera-large`)  
- `--overlay-dir`: Directory for overlay output (default: `target_overlay`)  
- `--mask-dir`: Directory for binary mask output (default: `target_binary_mask`)  
  
```bash  
python sam2_segmentation.py <image_path> [--model MODEL] [--overlay-dir DIR] [--mask-dir DIR]  
```  

**Example:**  
```bash  
python sam2_segmentation.py images/building.jpg --overlay-dir output/overlays  
```  
  
### 2. Patch Cropping  
  
Crop a smaller patch from the segmented area for processing.  
  
**Required Arguments:**  
- `image`: Path to input image  
- `mask`: Path to binary mask  
  
**Optional Arguments:**  
- `--out_dir`: Output directory (default: current directory)    
- `--crop_size`: Size of the square crop patch (default: `768`)  
  
```bash  
python crop.py <image> <mask> [--out_dir DIR] [--crop_size SIZE]  
```  
  
**Example:**  
```bash  
python crop.py images/building.jpg target_binary_mask/building_mask.png --crop_size 768  
```  
  
### 3. Crack Mask Generation  
  
Generate random binary crack patterns based on the cropped target region.  
  
**Required Arguments:**  
- `input`: Path to input cropped target region binary mask  
  
**Optional Arguments:**  
- `--num_cracks`: Number of crack seeds (default: `3`)  
- `--min_length`: Minimum crack length in pixels (default: `300`)  
- `--max_length`: Maximum crack length in pixels (default: `500`)  
- `--branch_prob`: Branching probability (default: `0.3`)  
- `--thickness_scale`: Thickness scale (default: `1.0`)  
  
```bash  
python crack_mask_generator.py <input> [OPTIONS]  
```  
  
**Example:**  
```bash  
python crack_mask_generator.py cropped_mask.png --num_cracks 5 --max_length 600 --branch_prob 0.4  
```  
  
### 4. Crack Image Generation  
  
Use ControlNet-guided Stable Diffusion to generate photorealistic cracks.  
  
**Required Arguments:**  
- `image_path`: Path to the cropped original image  
- `mask_path`: Path to the crack mask image  
  
**Optional Arguments:**  
- `--output_path`: Path to save output (default: `original_image_with_cracks.png`)  
- `--seed`: Random seed for reproducibility (default: `1`)  
- `--guidance_scale`: Guidance scale for prompt adherence (default: `70`)  
- `--controlnet_scale`: ControlNet conditioning scale (default: `2.5`)  
- `--inference_steps`: Number of inference steps (default: `200`)  
   
```bash  
python crack_generator.py <image_path> <mask_path> [OPTIONS]  
```  
  
**Example:**  
```bash  
python crack_generator.py cropped_image.png crack_mask.png --guidance_scale 80 --inference_steps 250  
```  
  
### 5. Full-Size Image Reconstruction  
  
Recreate full-size crack images by inserting the generated patch back into the original image.  
   
**Required Arguments:**  
- `original`: Path to the original full-size image  
- `patch`: Path to the generated crack patch  
- `patch_mask`: Path to the patch crack mask  
- `coords`: Path to JSON file with crop coordinates  
  
**Optional Arguments:**  
- `--output`: Path to save modified image (default: original filename with `_modified` suffix)  
- `--output_mask`: Path to save full-size crack mask (default: original filename with `_crack_mask` suffix)  
  
```bash  
python recreate.py <original> <patch> <patch_mask> <coords> [OPTIONS]  
```  
  
**Example:**  
```bash  
python recreate.py images/building.jpg output/crack_patch.png output/crack_mask.png coords.json  
```  
  
## Automation Scripts  
  
For batch processing, use the provided bash scripts:  
  
### Automatic Crack Mask Generation  
```bash  
bash auto_crack_mask_generation.sh  
```  
  
### Automatic Crack Image Generation  
```bash  
bash auto_crack_image_generation.sh  
```  
  
## Pipeline Workflow  
  
```  
Original Image
    ↓
[SAM2 Segmentation] → Target region mask + overlay
    ↓
[Crop] → 768×768 patches (image + mask)
    ↓
[Crack Mask Generator] → Synthetic crack mask
    ↓
[ControlNet + Stable Diffusion] → Patch with realistic cracks
    ↓
[Recreate] → Full-size image with cracks + full-size crack mask
```

## Output Structure  

```
project/
├── target_overlay/          # SAM2 segmentation overlays
├── target_binary_mask/      # SAM2 binary masks
├── cropped_images/          # Cropped patches
├── crack_masks/             # Generated crack masks
├── generated_cracks/        # Crack images (patches)
└── final_outputs/           # Full-size reconstructed images
```

## Requirements  

See `requirements.txt` for the complete list of dependencies. Key requirements include:  
- PyTorch  
- Transformers (Hugging Face)  
- Diffusers  
- SAM2  
- OpenCV  
- NumPy  
- PIL  

## Notes  

- The system works best with 768×768 patches due to the model's training resolution  
- Higher `guidance_scale` values produce cracks that adhere more closely to the mask  
- Adjust `controlnet_scale` to control the strength of ControlNet conditioning  
- Use consistent seeds for reproducible results  



