#!/bin/bash

# Path to your Python script
PYTHON_SCRIPT="crack_mask_generator.py"

# Path to your input binary mask
INPUT_MASK="001_cropped_mask.png"

# Number of images per length range
NUM_IMAGES=10

# Number of cracks per image (for filename)
NUM_CRACKS=3

# Parent output directory
OUTPUT_PARENT_DIR="generated_mask2"
mkdir -p "$OUTPUT_PARENT_DIR"

# Generate length ranges from 50-150 to 750-850 with interval of 100
LENGTH_RANGES=()
START=50
END=750
INTERVAL=100

for (( i=START; i<=END; i+=INTERVAL )); do
    RANGE_START=$i
    RANGE_END=$((i + INTERVAL))
    LENGTH_RANGES+=("${RANGE_START}-${RANGE_END}")
done

# Loop through each length range
for RANGE in "${LENGTH_RANGES[@]}"
do
    # Create a directory for this range inside parent folder
    DIR="$OUTPUT_PARENT_DIR/cracks_$RANGE"
    mkdir -p "$DIR"

    # Split min and max length, remove any extra commas/spaces
    MIN_LEN=$(echo $RANGE | cut -d'-' -f1 | tr -d ', ')
    MAX_LEN=$(echo $RANGE | cut -d'-' -f2 | tr -d ', ')

    # Generate NUM_IMAGES for this range
    for i in $(seq 1 $NUM_IMAGES)
    do
        # Run Python script
        python3 "$PYTHON_SCRIPT" \
            --input "$INPUT_MASK" \
            --num_cracks $NUM_CRACKS \
            --min_length $MIN_LEN \
            --max_length $MAX_LEN \
            --branch_prob 0.5 \
            --thickness_scale 2

        # Compute the Python output filename (handles double underscore)
        BASE_NAME=$(basename "$INPUT_MASK" .png)                  # '003_cropped_mask'
        OUTPUT_FILE="${BASE_NAME//cropped_mask/}_crack_mask.png"  # '003__crack_mask.png'

        # Move the generated image into the folder with unique name including num cracks
        mv "$OUTPUT_FILE" "$DIR/${BASE_NAME}_crack${NUM_CRACKS}_${MIN_LEN}_${MAX_LEN}_${i}.png"
        echo "Saved: $DIR/${BASE_NAME}_crack${NUM_CRACKS}_${MIN_LEN}_${MAX_LEN}_${i}.png"
    done
done
