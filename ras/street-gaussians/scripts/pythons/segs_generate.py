# Copyright 2026 Advanced Micro Devices, Inc.  All rights reserved.
 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
 
#       http://www.apache.org/licenses/LICENSE-2.0
 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import torch
import numpy as np
import cv2
import tqdm
from pathlib import Path
from PIL import Image
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

def get_parser():
    parser = argparse.ArgumentParser(description="Mask2Former semantic segmentation generation")
    parser.add_argument("--root_path", help="Root path containing 'images' folder")
    parser.add_argument("--model_name", default="facebook/mask2former-swin-large-mapillary-vistas-semantic", help="HuggingFace model ID")
    return parser

def get_files_in_folder(folder_path):
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                file_paths.append(os.path.join(root, file))
    return file_paths

if __name__ == "__main__":
    args = get_parser().parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    root_path = args.root_path
    base_dir = os.path.join(root_path, 'images')
    save_dir = os.path.join(root_path, 'segs')

    print(f"Loading Mask2Former model: {args.model_name}...")
    processor = Mask2FormerImageProcessor.from_pretrained(args.model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    file_paths = get_files_in_folder(base_dir)
    label_save_paths = [file_path.replace(base_dir, save_dir) for file_path in file_paths]

    for labelpath in label_save_paths:
        dirname = os.path.dirname(labelpath)
        if not os.path.exists(dirname): 
            os.makedirs(dirname)

    print(f"Processing {len(file_paths)} images...")

    for i in tqdm.tqdm(range(len(file_paths))):
        src_path = file_paths[i]
        dst_path = Path(label_save_paths[i]).with_suffix(".png").as_posix()
        if os.path.exists(dst_path):
            continue

        image = Image.open(src_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt", do_resize=False).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            
            predicted_semantic_map = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image.size[::-1]]
            )[0]
            
            pred_train_id = predicted_semantic_map.cpu().numpy().astype(np.uint8)

            cv2.imwrite(dst_path, pred_train_id)

    print("Done.")