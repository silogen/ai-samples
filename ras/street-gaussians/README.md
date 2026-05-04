<div align="center">
<h3 style="font-size:2.0em;">Street-Gaussians-ns</h3>
</div>
<div align="center">

[Introduction](#introduction) ·
[Installation and Run](#installation-and-run) ·
[Acknowledgements](#acknowledgements) ·

</div>

<p align="center">
  <video width="100%" autoplay muted>
    <source src="assets/render_grid.mp4" type="video/mp4">
  </video>
</p>


# Introduction

This is a fork of [the unofficial implementation of Street Gaussians](https://github.com/LightwheelAI/street-gaussians-ns/), which enables Street Gaussians on AMD GPUs.

It modifies the original codebase in the following ways:
* Makes Street Gaussians work with a more recent version of **gsplat**, which requires major changes to the
  way the densification process.
  * Check out gsplat migration [here](https://docs.gsplat.studio/main/migration/migration_legacy.html#basic-usage).
* Allows Street Gaussians to be run on both AMD and NVIDIA GPUs, mainly thanks to the gsplat upgrade,
  which means we can run with AMD-enabled gsplat.
* Adds some new options to allow experimentation with some new techniques and to output more debugging
  images and metrics.
* Refactors the code of the models so that it is easier to understand how the loss is computed, in particular
  what computation is done on the scene *graph* (the union of the background and object models) and what is
  done on the individual models.
* Enhances the rendering tool to render new videos from a trained scene:
  * Interpolation between training poses to generate smooth videos at a higher framerate than the training input.
  * Some simple editing operations, specified using config files, that manipulate the paths of the dynamic objects
    to produce a new variant of the scenario.


## Installation and Run

1. **Download a scene** from Waymo dataset
    1. Go [here](https://waymo.com/open/download/) and login with your google account and complete the registration form.
    2. Download [waymo_open_dataset_v_1_4_0](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_0) -> individual_files -> validation -> segment_1906113358876584689 and add it to the data directory.

2. **Build Docker images:** build the main and the data pre-processing docker images:
    ```sh
    docker build -t street-gaussians .
    docker build -t street-gaussians-data-proc -f Dockerfile.data_proc .
    ```

3. **Data preprocessing:** inside the data pre-processing docker container:
    First extract the data from the Waymo dataset using the Waymo tool (script from Street Gaussians).
    ```sh
    python scripts/pythons/extract_waymo.py --waymo_root ./data --out_root ./data --split validation
    ```
    Street Gaussians provides us with a data processing script that calls these scripts:
    * segs_generate.sh: Semantic segmentation of images using Mark2Former, which will be used for creating sky masks.
    * masks_generate.sh: create dynamic object masks, which will be used in COLMAP to avoid feature extraction in those regions.
    * run_colmap.sh: Run the COLMAP pipeline to generate an initial point cloud.
    * points_cloud_generate.sh: Get a point cloud also from lidar data and combine it with the COLMAP point cloud.
    * object_pts_generate.sh: Load the point cloud and extract regions corresponding to detected objects into separate clouds.
    
    In the original dataset, the images for each camera are stored in different subfolders.

    ```sh
    bash -e scripts/shells/data_process.sh ./data/validation/1906113358876584689_1359_560_1379_560/
    ```

    Exit the data pre-processing container.

6. **Run training:** inside the main docker container:
    ```sh
    bash scripts/shells/train/train.sh ./data/validation/1906113358876584689_1359_560_1379_560/
    ```

    Check the parameters that you can modify:
    ```sh
    sgn-train -h
    sgn-train street-gaussians-ns -h
    sgn-train street-gaussians-ns colmap-data-parser-config -h
    ```

    **NOTE:** the default parameters are set in `sgn_config.py` script and differ somewhat from the original codebase.

7. **Run rendering:** inside the base container:
    ```sh
    bash scripts/shells/render.sh [config_file]
    ```

    Check the perameters that you can modify:
    ```sh
    sgn-render -h
    ```

8. **Perform scene editing:** inside the base container:
    ```sh
    bash scripts/shells/render.sh [config_file] [edit_config_file]
    ```

8. **Run evaluation:** inside the base container:
    ```sh
    bash scripts/shells/eval.sh [config_file]
    ```

    Check the perameters that you can modify:
    ```sh
    sgn-eval -h
    ```


# Acknowledgements

## Built On

- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio/tree/main), a collaboration friendly studio for NeRFs.
- [Unofficial implementation of Street Gaussians](https://github.com/LightwheelAI/street-gaussians-ns/) from which this codebase was forked.
- [Official implementation of Street Gaussians](https://github.com/zju3dv/street_gaussians) by the paper's authors, used as a reference for some of the fixes in this codebase.
- [gsplat](https://github.com/ROCm/gsplat), a GPU-optimized 3D gaussian splatting library, with AMD GPU computation enabled.
- The video in the assets directory was made using the Waymo Open Dataset, provided by Waymo LLC under the Waymo Dataset License Agreement for Non-Commercial Use, available at https://www.waymo.com/open/terms. Access to and use of the dataset and any derivative works are governed by the terms and conditions therein.

## Citation
If you find this code useful, please be so kind to cite
```
@inproceedings{yan2024street,
    title={Street Gaussians for Modeling Dynamic Urban Scenes}, 
    author={Yunzhi Yan and Haotong Lin and Chenxu Zhou and Weijie Wang and Haiyang Sun and Kun Zhan and Xianpeng Lang and Xiaowei Zhou and Sida Peng},
    booktitle={ECCV},
    year={2024}
}
```