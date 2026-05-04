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

"""
Street Gaussians configuration file.
"""
from pathlib import Path

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from street_gaussians_ns.data.sgn_datamanager import FullImageDatamanagerConfig
from street_gaussians_ns.data.sgn_dataparser import ColmapDataParserConfig
from street_gaussians_ns.data.utils.bbox_optimizers import BBoxOptimizerConfig
from street_gaussians_ns.sgn_component_model import StreetGaussiansComponentModelConfig
from street_gaussians_ns.sgn_scene_graph import StreetGaussiansGraphModelConfig


street_gaussians_ns_method = MethodSpecification(
    config=TrainerConfig(
        method_name="street-gaussians-ns",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=30000, 
        max_num_iterations=30000,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 100,'semantic':10},
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=ColmapDataParserConfig(
                    data=Path(),
                    scale_factor=1.0,
                    downscale_factor=None,
                    scene_scale=1.0,
                    orientation_method="up",
                    center_method="poses",
                    auto_scale_poses=True,
                    assume_colmap_world_coordinate_convention=False,
                    eval_mode="interval",
                    train_split_fraction=0.9,
                    eval_interval=8,
                    filter_camera_id=[1],
                    images_path=Path("images"),
                    masks_path=None,
                    segments_path=Path("segs"),
                    colmap_path=Path("colmap/sparse/0"),
                    init_points_filename="points3D_withlidar.bin",
                    meta_file=Path("transform.json"),
                    load_3D_points=True,
                    max_2D_matches_per_3D_point=0,
                    undistort=True,
                    force_save_undistort_data=(),
                    load_dynamic_annotations=True,
                    frame_select=None,
                ),
                camera_res_scale_factor=1.0,
                eval_num_images_to_sample_from=-1,
                eval_num_times_to_repeat_images=-1,
                eval_image_indices=(0,),
                cache_images="cpu",
                cache_images_type="float32",
                max_thread_workers=None,
            ),
            model=StreetGaussiansGraphModelConfig(
                bbox_optimizer=BBoxOptimizerConfig(mode="simple"),
                object_acc_entropy_loss_mult=0,
                stats_every=0,
                absgrad=False,
                use_sky_sphere=True,
                env_map_res=1024,
                ssim_impl="pytorch-msssim",
                num_downscales=0,
                resolution_schedule=250,
                ssim_lambda=0.2,
                sky_acc_loss_mult=0.5,
                output_training_renders=0,
                background_model=StreetGaussiansComponentModelConfig(
                    warmup_length=500,
                    refine_every=100,
                    cull_alpha_thresh=0.02,
                    cull_scale_thresh=0.2,
                    continue_cull_post_densification=False,
                    reset_alpha_every=30,
                    densify_grad_thresh=0.0002,
                    densify_size_thresh=0.01,
                    n_split_samples=2,
                    cull_screen_size=0.15,
                    split_screen_size=0.05,
                    stop_screen_size_at=4000,
                    random_init=False,
                    num_random=50000,
                    random_scale=10.0,
                    stop_split_at=25000,
                    sh_degree=3,
                    fourier_features_dim=1,
                    fourier_features_scale=1,
                ),
                object_model_template=StreetGaussiansComponentModelConfig(
                    warmup_length=500,
                    refine_every=100,
                    cull_alpha_thresh=0.005,
                    cull_scale_thresh=0.2,
                    continue_cull_post_densification=False,
                    reset_alpha_every=30,
                    densify_grad_thresh=0.0002,
                    densify_size_thresh=0.01,
                    n_split_samples=2,
                    cull_screen_size=0.15,
                    split_screen_size=0.05,
                    stop_screen_size_at=4000,
                    random_init=False,
                    num_random=10000,
                    random_scale=10.0,
                    stop_split_at=25000,
                    sh_degree=3,
                    fourier_features_dim=5,
                    fourier_features_scale=1,
                )
            ),
        ),
        optimizers={
            "sky_sphere": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "bbox_opt":{
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            },
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer_legacy+tensorboard",
    ),
    description="Base config for Street Gaussians",
)