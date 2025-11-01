# gen_crack

# install the required library, run
"pip install -r requirements.txt"


# install SAM2, run
"git clone https://github.com/facebookresearch/sam2.git && cd sam2"
"pip install -e ."

# Download the SAM2 Models, run
"cd checkpoints && \
./download_ckpts.sh && \
cd .."

# Copy sam2_segmentation.py into the folder "sam2", run
"cp sam2_segmentation.py  sam2/sam2_segmentation.py"