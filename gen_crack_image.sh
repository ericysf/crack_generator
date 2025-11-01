#!/bin/bash
# ================================================================
# Configuration - UPDATE THESE PATHS
# ================================================================
ORIGINAL_IMAGES_DIR="/home/ubuntu/Desktop/code/20251020_to_vm/cropped_images_for_generation"
GENERATED_MASKS_DIR="/home/ubuntu/Desktop/code/20251020_to_vm/generated_mask1.5"
OUTPUT_DIR="/home/ubuntu/Desktop/code/20251020_to_vm/generated_crack"
PYTHON_SCRIPT="cn2.py"

# Optional: Override default parameters
SEED=1
GUIDANCE_SCALE=90
CONTROLNET_SCALE=3.0
INFERENCE_STEPS=200

# ================================================================
# Validation
# ================================================================
if [ ! -d "$ORIGINAL_IMAGES_DIR" ]; then
    echo "Error: Original images directory not found: $ORIGINAL_IMAGES_DIR"
    exit 1
fi

if [ ! -d "$GENERATED_MASKS_DIR" ]; then
    echo "Error: Generated masks directory not found: $GENERATED_MASKS_DIR"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# ================================================================
# Processing
# ================================================================
echo "Starting batch crack generation..."
echo "Original images: $ORIGINAL_IMAGES_DIR"
echo "Masks directory: $GENERATED_MASKS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Seed: $SEED"
echo "Guidance scale: $GUIDANCE_SCALE"
echo "ControlNet scale: $CONTROLNET_SCALE"
echo "Inference steps: $INFERENCE_STEPS"
echo "=================================================="

total_processed=0
total_failed=0

# Iterate through all crack size subfolders
for size_folder in "$GENERATED_MASKS_DIR"/cracks_*; do
    if [ ! -d "$size_folder" ]; then
        continue
    fi
    
    size_name=$(basename "$size_folder")
    echo "Processing: $size_name"
    
    # Iterate through all masks in the subfolder
    for mask_file in "$size_folder"/*_mask_*.png; do
        if [ ! -f "$mask_file" ]; then
            continue
        fi
        
        mask_basename=$(basename "$mask_file")
        
        # Extract original image name (e.g., "001_cropped" from "001_cropped_mask_crack3_50_150_1.png")
        # Pattern: {image_name}_mask_crack{number}_{size1}_{size2}_{iteration}.png
        image_name=$(echo "$mask_basename" | sed 's/_mask_crack.*//')
        original_image="$ORIGINAL_IMAGES_DIR/${image_name}.png"
        
        if [ ! -f "$original_image" ]; then
            echo "  ⚠ Warning: Original image not found: $original_image"
            ((total_failed++))
            continue
        fi
        
        # Generate output filename
        mask_iteration=$(echo "$mask_basename" | sed 's/.*_\([0-9]*\)\.png/\1/')
        output_image="$OUTPUT_DIR/${image_name}_${size_name}_crack_${mask_iteration}.png"
        
        echo "  Processing: $mask_basename -> $(basename "$output_image")"
        
        # Run the Python script with updated parameters
        if python "$PYTHON_SCRIPT" \
            "$original_image" \
            "$mask_file" \
            --output_path "$output_image" \
            --seed "$SEED" \
            --guidance_scale "$GUIDANCE_SCALE" \
            --controlnet_scale "$CONTROLNET_SCALE" \
            --inference_steps "$INFERENCE_STEPS"; then
            ((total_processed++))
            echo "  ✓ Success"
        else
            echo "  ✗ Error processing: $mask_basename"
            ((total_failed++))
        fi
    done
done

# ================================================================
# Summary
# ================================================================
echo "=================================================="
echo "Batch processing complete!"
echo "Total processed: $total_processed"
echo "Total failed: $total_failed"
echo "Output saved to: $OUTPUT_DIR"