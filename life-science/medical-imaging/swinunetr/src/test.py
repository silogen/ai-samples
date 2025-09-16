# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by Reference Models team (AMD) on 2025:
# - Adaptations for training on the NSCLCL-Radiomics dataset
# - Parameter changes
# - Logging and saving of standard output and error to file

import argparse
import json
import os
import sys
import traceback
from datetime import datetime as dt

import nibabel as nib
import numpy as np
import torch
from utils.data_utils import get_loader, LABEL_DATA, IMAGE_DATA
from utils.utils import dice
from utils.utils import Tee

from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_root_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="test1", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name",
    default="swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--download_data", action="store_true", help="Download the dataset")
parser.add_argument("--dataset_labels", default="GTV-1", type=str, help="One or more comma-separated labels (segmentation mask) to predict from the dataset")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=3, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-1000.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=1000.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=3.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="dropout path rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--save_results", action="store_true", help="Save results to logs folder")
parser.add_argument("--save_results_limit", default=15, type=int, help="Max number of results to save")


def main():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_stdout = None
    log_file_handle = None

    args = parser.parse_args()
    args.test_mode = True
    output_directory = os.path.join("./outputs", args.exp_name, dt.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_directory, exist_ok=True)

    # record input arguments and environemnt
    with open(os.path.join(output_directory, "args.json"), "wt") as fo:
        json.dump(vars(args), fo, indent=2)
    with open(os.path.join(output_directory, "env.json"), "wt") as fo:
        json.dump(dict(os.environ), fo, indent=2)

    viz_output_directory = os.path.join(output_directory, "visualizations")
    log_file_path = os.path.join(output_directory, "output.log")

    try:
        log_file_handle = open(log_file_path, "a", encoding="utf-8")  # 'a' for append
        tee_stdout = Tee(log_file_handle, original_stdout)
        tee_stderr = Tee(log_file_handle, original_stderr)  # Log stderr to the same file
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        print(f"Logging to: {log_file_path}")

        if args.save_results:
            os.makedirs(viz_output_directory, exist_ok=True)
            print(f"Visualization samples will be saved to: {viz_output_directory}")

        val_loader = get_loader(args)
        pretrained_dir = args.pretrained_dir
        model_name = args.pretrained_model_name
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pretrained_pth = os.path.join(pretrained_dir, model_name)

        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=args.dropout_path_rate,
        )

        model_dict = torch.load(pretrained_pth, weights_only=False)["state_dict"]
        model.load_state_dict(model_dict)
        model.eval()
        model.to(device)

        with torch.no_grad():
            dice_list_case = []
            skipped_batch_count = 0
            total_batches = len(val_loader)
            loader_iterator = iter(val_loader)
            saved_results_count = 0

            for i in range(total_batches):
                try:
                    batch = next(loader_iterator)
                    val_inputs, val_labels = (batch[IMAGE_DATA].cuda(), batch[LABEL_DATA].cuda())
                    img_name = batch[IMAGE_DATA].meta['filename_or_obj'][0].split("/")[-3]
                    print(f"{i}/{total_batches} - Inference on case {img_name}")
                    print(f"Shape inputs/labels: {val_inputs.shape} / {val_labels.shape}")
                    print(f"Inputs min/max: {val_inputs.min()}/{val_inputs.max()}")
                    _, _, h, w, d = val_labels.shape

                    image_affine_tensor = batch[IMAGE_DATA].affine[0]
                    label_affine_tensor = batch[LABEL_DATA].affine[0]
                    image_affine_np = image_affine_tensor.cpu().numpy()
                    label_affine_np = label_affine_tensor.cpu().numpy()

                    val_outputs = sliding_window_inference(
                        val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
                    )

                    assert val_inputs.shape[0] == val_labels.shape[0] == 1, "Batch size on test script should be 1"

                    val_inputs = val_inputs[0]
                    val_outputs = val_outputs[0]
                    val_labels = val_labels[0]

                    print(f"val_inputs shape: {val_inputs.shape}")
                    print(f"val_outputs shape: {val_outputs.shape}")
                    print(f"val_labels shape: {val_labels.shape}")

                    val_outputs_probs = torch.sigmoid(val_outputs).cpu().numpy()
                    val_outputs = (val_outputs_probs > 0.5).astype(np.uint8)
                    val_labels = val_labels.cpu().numpy()

                    if args.out_channels == 1:
                        organ_Dice = dice(val_outputs[0], val_labels[0])
                        print("Organ Dice: {}".format(organ_Dice))
                        dice_list_case.append(organ_Dice)
                    else:
                        dice_list_sub = []
                        for ch in range(args.out_channels):
                            organ_Dice = dice(val_outputs[ch], val_labels[ch])
                            dice_list_sub.append(organ_Dice)

                        mean_dice = np.mean(dice_list_sub)
                        print("Mean Organ Dice: {}".format(mean_dice))
                        dice_list_case.append(mean_dice)

                    if args.save_results and saved_results_count < args.save_results_limit:

                        save_inputs = val_inputs.cpu().numpy().transpose(1, 2, 3, 0)
                        save_labels = val_labels.transpose(1, 2, 3, 0)
                        save_outputs = val_outputs.transpose(1, 2, 3, 0)
                        save_probs = val_outputs_probs.astype(np.float32).transpose(1, 2, 3, 0)

                        if args.out_channels == 1:
                            save_inputs = save_inputs.squeeze()

                        nib.save(
                            nib.Nifti1Image(save_inputs, image_affine_np),
                            os.path.join(viz_output_directory, f"{img_name}_input_channel0.nii.gz")
                        )
                        nib.save(
                            nib.Nifti1Image(save_labels, label_affine_np),
                            os.path.join(viz_output_directory, f"{img_name}_groundtruth_label.nii.gz")
                        )
                        nib.save(
                            nib.Nifti1Image(save_outputs, label_affine_np),
                            os.path.join(viz_output_directory, f"{img_name}_prediction_binary.nii.gz")
                        )
                        nib.save(
                            nib.Nifti1Image(save_probs, label_affine_np),
                            os.path.join(viz_output_directory, f"{img_name}_prediction_probabilities.nii.gz")
                        )
                        print(f"Visualization saved to {viz_output_directory}")

                        saved_results_count += 1

                except Exception as e:
                    print(f"ERROR loading batch at index {i}. Skipping this batch.")
                    print(f"Error details: {e}")
                    print(traceback.format_exc())
                    skipped_batch_count += 1
                    continue

            print(f"Skipped batches: {skipped_batch_count}")
            print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))

    except Exception as e:
        print(f"Error details: {e}")
        print(traceback.format_exc())
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if tee_stdout:
            tee_stdout.close()
        elif log_file_handle:
            log_file_handle.close()


if __name__ == "__main__":
    main()
