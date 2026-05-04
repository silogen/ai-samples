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

from __future__ import annotations

import json
import copy
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from torch.nn import Parameter
import torchvision.transforms.functional as TF

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from gsplat import rasterization

from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps

from street_gaussians_ns.sgn_component_model import StreetGaussiansComponentModel, StreetGaussiansComponentModelConfig
from street_gaussians_ns.data.utils.bbox_optimizers import BBoxOptimizerConfig, BBoxOptimizer
from street_gaussians_ns.data.utils.dynamic_annotation import InterpolatedAnnotationSet
from street_gaussians_ns.rendering import get_viewmat, render_outputs, render_empty
from street_gaussians_ns.sky import EnvLight
from street_gaussians_ns.utils import object2world_gs
from street_gaussians_ns.data.utils.data_utils import SemanticType
from street_gaussians_ns.data.cameras import Cameras


torch.compiler.allow_in_graph(rasterization)
# Without this we get:
#  TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. 
#  Consider setting `torch.set_float32_matmul_precision('high')` for better performance
torch.set_float32_matmul_precision("high")
# Tell the torch compiler to allow dynamic shapes: input shapes vary when we do refinement
torch._dynamo.config.force_parameter_static_shapes = False

def get_ssim(ssim_impl="pytorch-msssim"):
    if ssim_impl == "fused-ssim":
        warnings.warn("Using fused-ssim for SSIM calculation: this does not currently produce the same results as PyTorch-MSSSIM")
        from fused_ssim import fused_ssim
        # NOTE: currently this does not produce the same results as pytorch-msssim
        # Use padding="valid" for compatibility with pytorch-msssim
        # fused-ssim is fixed to data_range=1
        # It also always does the averaging that pytorch-msssim does with size_average=True
        return lambda X, Y: fused_ssim(X, Y, padding="valid")
    else:
        from pytorch_msssim import SSIM
        return SSIM(data_range=1.0, size_average=True, channel=3)


@dataclass
class StreetGaussiansGraphModelConfig(ModelConfig):
    """Gaussian Splatting Model Config"""

    _target: Type = field(default_factory=lambda: StreetGaussiansGraphModel)

    background_model: StreetGaussiansComponentModelConfig = field(default_factory=StreetGaussiansComponentModelConfig)
    """Background model config"""
    object_model_template: StreetGaussiansComponentModelConfig = field(default_factory=StreetGaussiansComponentModelConfig)
    """Object model config"""
    bbox_optimizer: BBoxOptimizerConfig = field(default_factory=BBoxOptimizerConfig)
    """Bounding box optimizer config"""
    object_acc_entropy_loss_mult: float = 0.001
    """loss weight of object-background accumulation cross entropy loss"""
    stats_every : int = 0
    """Output training stats every N iterations (default=0, don't output stats)"""
    absgrad : bool = False
    """Use absolute 2D gradients (computed using gsplat's absgrad option) as the high-grad criterion for refinement"""
    use_sky_sphere: bool = False
    """Enable sky sphere in rendering."""
    env_map_res: int = 1024
    """Resolution of the environment map used for the sky sphere."""
    ssim_impl: Literal["pytorch-msssim", "fused-ssim"] = "pytorch-msssim"
    """Implementation of SSIM loss to use. If you choose fused-ssim, you must have the fused_ssim library installed. Default: pytorch-msssim (original Street Gaussians implementation)"""
    num_downscales: int = 0
    """at the beginning, resolution is 1/2^d, where d is this number"""
    resolution_schedule: int = 250
    """training starts at 1/d resolution, every n steps this is doubled"""
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    sky_acc_loss_mult: float = 0.5
    """weight of sky accumulation loss"""
    output_training_renders: int = 0
    """Outputs a lot of images during training for debugging purposes. 
    Slows down training, but gives useful insight into the training process.
    Parameter controls how many camera poses are output"""


class StreetGaussiansGraphModel(Model):
    """Gaussian Splatting model

    Args:
        config: Gaussian Splatting configuration to instantiate model
    """
    config: StreetGaussiansGraphModelConfig

    def __init__(self,*args,**kwargs,):
        super().__init__(*args, **kwargs)
        self.render_info = None
    
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = get_ssim(self.config.ssim_impl)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        
        self.debug_cams = []
        self.log_dir = None

    def to(self, device):
        obj: StreetGaussiansGraphModel = super().to(device)
        obj.bbox_optimizer = obj.bbox_optimizer.to(device)
        for timestamp, box in obj.object_annos.iter_boxes():
            box.to(device)
        return obj

    @property
    def num_points(self):
        return sum(om.num_points for om in self.all_models.values())
    
    @property
    def sh_degree(self):
        # sh_degree is forced to be the same on all models
        return self.background_model.config.sh_degree
    
    def get_object_model_name(self, object_id):
        return f"object_{object_id}"

    @property
    def background_model(self) -> StreetGaussiansComponentModel:
        return self.all_models["background"]

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = {}
        for model in self.all_models.values():
            for group_name, params in model.get_gaussian_param_groups().items():
                if group_name in groups:
                    groups[group_name] += params
                else:
                    groups[group_name] = params
        assert len(set(len(v) for v in groups.values())) == 1, "Submodules contain different gaussian param groups"
        return groups

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = self.get_gaussian_param_groups()
        # add bbox optimizer param groups
        self.bbox_optimizer.get_param_groups(groups)
        if self.config.use_sky_sphere:
            groups["sky_sphere"] = list(self.env_map.parameters())
        return groups

    def step_cb(self, step):
        """Callback at start of every step to allow other methods to access step count"""
        self.step = step

    def set_log_dir_cb(self, trainer, step):
        """Callback at start of training to give access to log dir for debugging output"""
        self.log_dir = trainer.base_dir / trainer.config.logging.relative_log_dir

    def populate_modules(self):
        self.seed_points = None
        
        # Fix some parameters to be the same on the object models as on the background
        for param_name in ["sh_degree"]:
            setattr(self.config.object_model_template, param_name, getattr(self.config.background_model, param_name))

        self.all_models = torch.nn.ModuleDict()
        self.all_models["background"] = self.config.background_model.setup(
            scene_box=self.scene_box,
            num_train_data=self.num_train_data,
            model_idx_in_scene_graph=0,
            **self.kwargs
        )

        self.object_annos: InterpolatedAnnotationSet = self.kwargs["metadata"].get("object_annos", InterpolatedAnnotationSet())
        for idx, (obj_id, obj_annos) in enumerate(self.object_annos.trajectories.items()):
            object_model_config = copy.deepcopy(self.config.object_model_template)
            object_name = self.get_object_model_name(obj_id)
            self.all_models[object_name] = object_model_config.setup(
                scene_box=self.scene_box,
                num_train_data=self.num_train_data,
                seed_points=obj_annos.seed_pts,
                metadata=self.kwargs["metadata"],
                model_idx_in_scene_graph = idx + 1,
                extent=obj_annos.meta.size / 2.,
            )
        
        self.bbox_optimizer: BBoxOptimizer = self.config.bbox_optimizer.setup(
            num_frames=len(self.object_annos.all_timestamps), 
            num_bboxes=len(self.object_annos.unique_track_ids), 
            device="cpu",
            bbox_list=self.object_annos.unique_track_ids,
        )

        if self.config.use_sky_sphere:
            # this implementation is from https://github.com/fudan-zvg/PVG/blob/main/train.py#L51
            self.env_map = EnvLight(resolution=self.config.env_map_res).cuda()

        self.stop_refinement_at = max(
            self.config.background_model.stop_split_at, 
            self.config.object_model_template.stop_split_at, 
            )

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        
        if self.config.output_training_renders > 0 or self.config.stats_every > 0:
            # If storing stats or debugging renders, get the output directory from the trainer
            cbs.append(TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], 
                self.set_log_dir_cb, 
                args=[training_callback_attributes.trainer],
                iters=[0],
            ))

        cbs.append(TrainingCallback(
            [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        
        if self.config.stats_every > 0:
            cbs.append(TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], 
                self.output_stats, 
                update_every_num_iters=self.config.stats_every
            ))

        # after_train sends stats needed for refinement to the sub-models
        cbs.append(TrainingCallback(
            [TrainingCallbackLocation.AFTER_TRAIN_ITERATION], self.after_train))
        
        if self.config.output_training_renders > 0:
            # Output an image before refinement
            cbs.append(
                TrainingCallback(
                    [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    self.output_debugging_image_callback,
                    update_every_num_iters=self.background_model.config.refine_every,
                    kwargs=dict(before=True),
                )
            )

        # Sub-models also need to run their callbacks to run refinement
        for model in self.all_models.values():
            cbs += model.get_training_callbacks(training_callback_attributes)

        if self.config.output_training_renders > 0:
            # Output an image after refinement
            cbs.append(
                TrainingCallback(
                    [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    self.output_debugging_image_callback,
                    update_every_num_iters=self.background_model.config.refine_every,
                    kwargs=dict(before=False),
                )
            )

        return cbs

    def output_stats(self, step: int):
        """
        Output some logging stats between iterations to track the model training.
        
        """
        if step == 0:
            return
        assert self.log_dir is not None, "log_dir was not set by callback"

        stats_filename = f"stats_{step:05d}.json"

        # Compile model stats
        model_stats = dict(
            (model_name, dict({
                "num_gaussians": int(model.means.shape[0]),
                "max_updated_gs": int(max(model._num_updated_gs)) if len(model._num_updated_gs) else None,
                "num_updates": len(model._num_updated_gs),
                "mean_updated_gs": (float(sum(model._num_updated_gs)) / len(model._num_updated_gs)) if len(model._num_updated_gs) else None,
                "num_visible_gs": torch.count_nonzero(model.vis_counts).item() if model.vis_counts is not None else None,
                "mean_opacity": model.opacities.mean().item(),
                "min_opacity": model.opacities.min().item(),
                "max_opacity": model.opacities.max().item(),
                "avg_scales_exp": model.scales.exp().mean(dim=-1).nanmean().item(),
                "max_scales_exp": model.scales.exp().mean(dim=-1).max().item(),
            }, **model._refine_record_dict_accum)) 
            for model_name, model in self.all_models.items()
        )
        # Clear accumulators to start over
        for model in self.all_models.values():
            model._num_updated_gs = []
            model._refine_record_dict_accum = {}

        with (self.log_dir / stats_filename).open("w") as f:
            json.dump({
                "models": model_stats,
            }, f)

    def after_train(self, step: int):
        """
        Called after the backward pass on every step. At this point, gradients are available,
        so we can accumulate the stats needed for refinement. Stats are split to the separate
        sub-models and accumulated on their instances by calling `update_refinement_stats()`.

        Everything we need to compute these stats has been made available in `self.render_info`
        by the `get_outputs()` call during rendering.

        """
        # Grads are now available for the concatenated xys, since we set retain_grad()
        # Split them up and store them on the separate models, so they can be used for refinement
        assert self.render_info is not None, "render_info did not get set"

        # To save some training time, we don't need to update separate models' refinement stats after refinement has stopped
        if self.step < self.stop_refinement_at:
            # Migration: The projected gaussian params we retrieve from meta are only for those gaussians that survived the
            #  culling, i.e. will be visible in this rasterized image
            # We therefore have to keep track of which guassians we are updating stats for when we collect model-level
            #  stats that will be used by refinement. This is stored in survivor_mask
            image_size = self.render_info["image_size"]

            if "radii" in self.render_info:
                survivor_mask = (self.render_info["radii"] > 0.).any(dim=-1).squeeze(0)
                # After the backward pass on the scene graph model, we will split the gradients of this to the separate models
                if self.config.absgrad:
                    grad = self.render_info["means2d"].absgrad.detach().clone()
                else:
                    grad = self.render_info["means2d"].grad.detach().clone()
                #grad[..., 0] *= image_size[1] / 2.0
                #grad[..., 1] *= image_size[0] / 2.0
                grad = grad.squeeze(0)
                # Migration: In previous version of gsplat the shape of radii was [N, 1], because it was used a single, maximum radius in pixels for the projected 2D Gaussian. 
                # The newer version uses anisotropic radii, i.e., [rx, ry], so the shape is [N, 2]. To keep consistent with previous version, we take the maximum radius here
                radii = self.render_info['radii'].max(dim=-1).values.detach().squeeze(0)
            else:
                # No gaussians survived
                survivor_mask = torch.zeros((0,), dtype=torch.bool)
                grad = torch.zeros((0, 2), dtype=torch.float32)
                radii = torch.zeros((0,), dtype=torch.int32)

            # Split up grads and radii tensors into separate models' portions
            start_id = 0
            for model_name in self.render_info["visible_model_names"]:
                model = self.all_models[model_name]
                num_points = model.num_points
                # Get the survivor IDs that apply to this model
                model_survivor_mask = survivor_mask[start_id:start_id+num_points, ...]
                grad_t = grad[start_id:start_id+num_points, ...]
                radii_t = radii[start_id:start_id+num_points, ...]

                # Update the stats mantained for refinement
                model.update_refinement_stats(model_survivor_mask, grad_t, radii_t, image_size)

                start_id += num_points
            
            # Clear the memory for the 2D intermediate tensors we stored
            if self.config.absgrad:
                del self.render_info['means2d'].absgrad
            else:
                # This is extremely important, as this memory doesn't get freed if we don't do this
                # It's needed because we called self.render_info['means2d'].retain_grad()
                del self.render_info['means2d'].grad
            del self.render_info['means2d']

    def output_debugging_image(self, output_path: Path, name: str, camera: Cameras, override_colors=None):
        """
        May be called between training iterations to output images of the separate
        models. The background model is rendered from an arbitrarily chosen camera
        pose. Other models are rendered side-on in the local space.

        """
        assert self.debug_cams is not None and len(self.debug_cams), "debug_cams not yet set"

        with torch.no_grad():
            for model_name, model in self.all_models.items():
                model_dir = output_path / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                file_path = model_dir / name

                if model_name != "background":
                    # For dynamic models, compute a camera position where we can see the model well
                    # For the background model, use the original camera position

                    # Start from origin
                    camera_to_worlds = torch.zeros((3, 4), dtype=torch.float32, device=camera.camera_to_worlds.device)[None, ...]
                    # No rotation
                    camera_to_worlds[0, 0, 0] = 1.
                    camera_to_worlds[0, 1, 1] = 1.
                    camera_to_worlds[0, 2, 2] = 1.
                    # These translations are inverted, as we're specifying the cam2world (not world2cam)
                    # Shift so the model is centred in x-y plane
                    camera_to_worlds[0, 0, 3] = -0.5 * (model.means[:, 0].min() + model.means[:, 0].max())
                    camera_to_worlds[0, 1, 3] = -0.5 * (model.means[:, 1].min() + model.means[:, 1].max())
                    #camera.camera_to_worlds[0, 2, 3] = model.means[:, 2].min() - (model.means[:, 2].max() - model.means[:, 2].min())
                    # Step back by the size of the model*backsteps in the z-plane
                    backsteps = 3.
                    camera_to_worlds[0, 2, 3] = backsteps*(model.means[:, 2].max() - model.means[:, 2].min()) - model.means[:, 2].min()
                    camera = Cameras(
                        camera_to_worlds=camera_to_worlds,
                        fx=camera.fx,
                        fy=camera.fy,
                        cx=camera.cx,
                        cy=camera.cy,
                        width=camera.width,
                        height=camera.height,
                        distortion_params=camera.distortion_params.detach().clone(),
                        camera_type=camera.camera_type,
                        times=camera.times.detach().clone(),
                        metadata=copy.deepcopy(camera.metadata),
                    )

                model.output_debugging_image_for_params(file_path, camera, override_colors=override_colors)

    def output_debugging_image_callback(self, step: int, before=True):
        """Callback to output an image either before or after updates."""
        assert self.log_dir is not None, "log_dir should have been set at the start of training"
        assert self.debug_cams is not None and len(self.debug_cams), "debug_cams not yet set"

        source_cam = self.debug_cams[0]
        if step == 0 and before:
            # Set the debug cam on all the models, so they can output images *during* refinement
            for model_name, model in self.all_models.items():
                # Make this camera and the output dir available to the model so it can output to it during refinement
                model_dir = self.log_dir / model_name
                model.debugging_cam = (source_cam, model_dir)
            
        name = "0_before_refine" if before else "3_after_refine"
        output_name = f"debug_render_{step:04d}_{name}.png"
        self.output_debugging_image(self.log_dir, output_name, source_cam)

    def collect_gaussians(
            self, 
            camera: Cameras, 
            show_models: Optional[List[str]] = None, 
            transform_objects: bool = True, 
            compute_mask: bool = False, 
            K_obj=None, 
            cam2world=None, 
            height_obj=None, 
            width_obj=None
    ):
        """
        Gather all the gaussian parameters for the object models that are visible at
        the timestamp of the given camera.
        
        :param camera: camera to get timestamp from
        :type camera: Cameras
        :param show_models: restrict parameters to the models in the list
        :type show_models: Optional[List[str]]
        """
        all_object_means = []
        all_object_quats = []
        all_object_features_dc = []
        all_object_scales = []
        all_object_opacities = []
        all_object_features_rest = []
        visible_models = []
        if not compute_mask:
            objects_mask = None
        else:
            # Initialize objects_mask as a tensor of zeros if compute_mask is True
            objects_mask = torch.zeros((height_obj, width_obj), dtype=torch.bool, device=self.device)

        # Get annotations for all visible objects for this timestamp
        # If training, only get actual annotations that exist in the set
        # If not, get (potentially interpolated) annotations for all objects that are visible at this time
        annos_for_time = self.object_annos.get_boxes_for_time(camera.time_ms, interpolate=not self.training)

        for track_id, anno in annos_for_time:
            model_name = self.get_object_model_name(track_id)
            obj_traj = self.object_annos.trajectories[track_id]

            if show_models is not None and model_name not in show_models:
                # This model is hidden
                continue
            
            if self.training and camera.time_ms in obj_traj.timestamps and self.bbox_optimizer.config.mode != "off":
                # Skip apply_to_bbox if not optimizing, because this adds a lot of unnecessary computation
                anno = self.bbox_optimizer.apply_to_bbox(anno, track_id, self.object_annos.timestamp_to_frame_num(camera.time_ms))

            if compute_mask:
                import cv2
                import numpy as np

                assert anno.size is not None
                assert cam2world is not None
                # Get the vertices of the bbox in world space (all on GPU)
                dx = torch.tensor([-1, -1, -1, -1, 1, 1, 1, 1], device=self.device)*anno.size[0]/2
                dy = torch.tensor([-1, 1, 1, -1, -1, 1, 1, -1], device=self.device)*anno.size[1]/2
                dz = torch.tensor([-1, -1, 1, 1, -1, -1, 1, 1], device=self.device)*anno.size[2]/2
                vertices = torch.vstack((dx, dy, dz)).T.to(torch.float32)
                vertices = vertices @ anno.rot.T
                vertices = vertices + anno.center
                # Transform vertices from world space to camera space (OpenGL convention: Right, Up, Back)
                vertices = (vertices - cam2world[:3, 3]) @ cam2world[:3, :3]
                # Convert to OpenCV convention (Right, Down, Forward)
                vertices = vertices * torch.tensor([1, -1, -1], dtype=vertices.dtype, device=vertices.device)
                # Project to image plane (all on GPU)
                projected_hom = vertices @ K_obj.T
                valid_mask = projected_hom[:, 2] > 0.01
                if not valid_mask.any():
                    continue
                projected_hom = projected_hom[valid_mask]
                uv = projected_hom[:, :2] / projected_hom[:, 2:3]
                pts = uv.round().to(torch.int32)
                # Now move to CPU for OpenCV
                pts_cpu = pts.cpu().numpy()
                object_mask_np = np.zeros((height_obj, width_obj), dtype=np.uint8)
                if len(pts_cpu) > 0:
                    hull = cv2.convexHull(pts_cpu)
                    cv2.fillConvexPoly(object_mask_np, hull, 1)
                object_mask = torch.from_numpy(object_mask_np).bool().to(self.device)
                if object_mask.sum() >= 100:
                    objects_mask |= object_mask

            assert model_name not in visible_models, f"model {model_name} appeared twice at timestamp {camera.time_ms}"
            obj_model = self.all_models[model_name]
            # prevent empty object
            if obj_model.num_points == 0:
                continue

            # Add fourier features of time
            if obj_model.config.fourier_features_dim > 1:
                # Compute the fourier features for this point on the object's trajectory
                # ApearanceDuringEditing: this should be removed when we come up with a better solution for handling appearance changes during edits.
                # Use appearance_time if the annotation specifies it (e.g. for stopped vehicles)
                effective_time = anno.appearance_time if anno.appearance_time is not None else camera.time_ms
                trajectory_prop = float(effective_time - obj_traj.start_time_look) / (obj_traj.end_time_look - obj_traj.start_time_look)
                all_object_features_dc.append(obj_model.get_fourier_features(trajectory_prop))
            else:
                all_object_features_dc.append(obj_model.features_dc)

            # Aggregate all models properties for splatting
            visible_models.append(model_name)
            if transform_objects:
                obj_means, obj_quats = object2world_gs(obj_model.means, obj_model.quats, anno)
            else:
                obj_means, obj_quats = obj_model.means, obj_model.quats
            all_object_means.append(obj_means)
            all_object_quats.append(obj_quats)
            all_object_scales.append(obj_model.scales)
            # Opacities and features_rest are not transformed, so we just gather them from the objects
            all_object_opacities.append(obj_model.opacities)
            all_object_features_rest.append(obj_model.features_rest)

        return all_object_means, all_object_quats, all_object_scales, all_object_opacities, all_object_features_dc, all_object_features_rest, visible_models, objects_mask

    def get_outputs(self, camera: Cameras, show_models: Optional[List[str]] = None) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Camera and renders, returning a dictionary of outputs.

        This renders the combined scene, consisting of the background model and all of the object
        models, or a subset of the models, if `show_models` is used to select some.
        Each object model is projected to the position corresponding to its annotation at the
        timestamp associated with the camera.

        As a side-effect of this call during training, `self.render_info` is set to a dictionary
        containing all of the data needed to update the sub-models' stats that are accumulated
        during steps and then used for refinement.

        Args:
            camera: Input camera of viewpoint. This camera should have all the
                    needed information to compute the outputs.
            show_models: Optionally restrict to include only a subset of the graph's sub-models.

        Returns:
            Outputs of model (ie. rendered colors). Includes 'rgb', 'accumulation' (alpha), 
                'opacities' (per-gaussian opacity), 'scales' (per-gaussian 2D scale).
                If `sky_capture` is enabled, also includes 'sky' (sky-only rendering).
                If training and the conditions are met for including the object-accumulation
                regularization term in the loss, also includes 'object_acc'.
        """
        K, viewmat = get_viewmat(camera, 1, camera.camera_to_worlds)
        # Save some camera poses so we use the same ones each time for outputing training renders
        is_debug_step = False
        if self.training and self.config.output_training_renders > 0:
            if len(self.debug_cams) < self.config.output_training_renders:
                self.debug_cams.append(camera)
                is_debug_step = True
            else:
                debug_cam_ids = [c.metadata["cam_idx"] for c in self.debug_cams]
                is_debug_step = camera.metadata["cam_idx"] in debug_cam_ids

        if show_models is None:
            show_models = list(self.all_models.keys())
        show_background = "background" in show_models
        object_models = [m for m in show_models if m != "background"]

        if self.config.use_sky_sphere and show_background:
            # Render the sky
            # Leave the sky out if we're not rendering the background
            sky_capture = self.env_map(camera.width, camera.height, camera.cx, camera.cy, camera.fx, camera.fy, camera.camera_to_worlds[0, ...], self.training)
        else:
            sky_capture = None

        # Object accumulation regularization term is only included after refinement, so skip computing object_acc until then
        compute_object_stuff = is_debug_step or (
                self.training and
                self.config.object_acc_entropy_loss_mult > 0. and 
                self.step >= self.config.background_model.stop_split_at
        )
        if compute_object_stuff:
            # Render object-only accumulation at a lower resolution to save GPU compute.
            d_obj = max(1, getattr(self.config, "object_acc_downscale", 2))
            K_obj = K.clone()
            # scale intrinsics for downsampled render; K layout: [...,3,3] or [3,3]
            K_obj[..., 0, 0] /= d_obj
            K_obj[..., 1, 1] /= d_obj
            K_obj[..., 0, 2] /= d_obj
            K_obj[..., 1, 2] /= d_obj
            width_obj = max(1, camera.width // d_obj)
            height_obj = max(1, camera.height // d_obj)
        else:
            K_obj = None
            height_obj = None
            width_obj = None

        # Accumulate parameters from the submodels (objects), excluding the background
        object_means, \
            object_quats, \
            object_scales, \
            object_opacities, \
            object_features_dc, \
            object_features_rest, \
            visible_object_models, \
            obj_mask = self.collect_gaussians(camera, object_models, compute_mask=compute_object_stuff, K_obj=K_obj, cam2world=camera.camera_to_worlds[0], height_obj=height_obj, width_obj=width_obj)
        
        if not show_background and (len(object_means) == 0 or sum(obj_mn.shape[0] for obj_mn in object_means) == 0):
            # No gaussians are visible
            outputs, meta = render_empty(
                camera.width,
                camera.height, 
                self.device, 
                output_names=["rgb", "accumulation"] + ([] if self.training else ["depth"]),
            )
        else:
            # Combine parameters from background and the submodels that we're showing
            def _cat_params(bg_params, object_params):
                return torch.cat(([bg_params] if show_background else []) + object_params, dim=0)
            means = _cat_params(self.background_model.means, object_means)
            quats = _cat_params(self.background_model.quats, object_quats)
            features_dc = _cat_params(self.background_model.features_dc, object_features_dc)
            opacities = _cat_params(self.background_model.opacities, object_opacities)
            features_rest = _cat_params(self.background_model.features_rest, object_features_rest)
            scales = _cat_params(self.background_model.scales, object_scales)

            # Perform rasterization
            outputs, meta = render_outputs(
                camera.width,
                camera.height,
                K, 
                viewmat,
                means,
                quats,
                scales,
                opacities,
                features_dc,
                features_rest,
                sky_capture,
                absgrad=self.training and self.config.absgrad,
                sh_degree=self.sh_degree,
                output_names=["rgb", "accumulation"] + ([] if self.training else ["depth"]),
                training=self.training,
            )
                
            # Include the opacities and scales in the outputs, so they can be used to compute metrics
            outputs["opacities"] = opacities
            outputs["scales"] = scales
        
        if self.training and self.step < self.stop_refinement_at:
            # Store the information we need for updating stats after the backward pass needed for refinement
            self.render_info = meta
            self.render_info["visible_model_names"] = (["background"] if show_background else []) + visible_object_models
            self.render_info["image_size"] = (camera.height, camera.width)

            if not self.config.absgrad:
                self.render_info['means2d'].retain_grad()
        
        if sky_capture is not None:
            outputs["sky"] = sky_capture

        # Object accumulation regularization term is only included after refinement, so skip computing object_acc until then
        if compute_object_stuff:
            # Also compute the accumulation, just for the objects, excluding the background
            # This is included in the loss fn as a regularization term
            if len(object_means) == 0 or sum(obj_mn.shape[0] for obj_mn in object_means) == 0:
                object_acc_outputs, __ = render_empty(width_obj, height_obj, self.device, output_names=["accumulation"] + (["rgb"] if is_debug_step else ["rgb"]))
            else:
                object_acc_outputs, __ = render_outputs(
                    width_obj,
                    height_obj,
                    K_obj, 
                    viewmat,
                    torch.cat(object_means, dim=0),
                    torch.cat(object_quats, dim=0),
                    torch.cat(object_scales, dim=0),
                    torch.cat(object_opacities, dim=0),
                    torch.cat(object_features_dc, dim=0),
                    torch.cat(object_features_rest, dim=0),
                    output_names=["accumulation"] + (["rgb"] if is_debug_step else []),
                    sh_degree=self.sh_degree,
                    training=self.training,
                )
            outputs["object_acc"] = object_acc_outputs["accumulation"]
            if "rgb" in object_acc_outputs:
                outputs["object_rgb"] = object_acc_outputs["rgb"]

            # # This is not currently debugging and fully working, so don't use it for now
            # # Instead, we follow the unofficial SG implementation by not using this mask
            outputs["obj_mask"] = obj_mask

        return outputs

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.

        Overridden for a camera-based gaussian model.

        Does not currently use the obb_box if given. (It was also ignored for the scene graph model in the original SG.)

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        losses = {}
        d = self._get_downscale_factor()
        mask = None
        gt_semantic = None

        if self.training and d > 1:
            newsize = (int(math.ceil(batch["image"].shape[0] / d)), int(math.ceil(batch["image"].shape[1] / d)))
            gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
            if "mask" in batch:
                mask = TF.resize(
                    batch["mask"].permute(2, 0, 1),
                    newsize,
                    antialias=None,
                    interpolation=TF.InterpolationMode.NEAREST,
                ).permute(1, 2, 0)
            if "semantic" in batch:
                gt_semantic = TF.resize(
                    batch["semantic"].permute(2, 0, 1),
                    newsize,
                    antialias=None,
                    interpolation=TF.InterpolationMode.NEAREST
                ).permute(1, 2, 0)
        else:
            gt_img = batch["image"]
            if "mask" in batch:
                mask = batch["mask"]
            if "semantic" in batch:
                gt_semantic = batch["semantic"]

        # RGB loss
        rgb = outputs["rgb"]
        if mask is not None:
            gt_img *= mask
            rgb *= mask
        Ll1 = torch.abs(gt_img - rgb).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], rgb.permute(2, 0, 1)[None, ...])
        losses["Ll1"] = (1 - self.config.ssim_lambda) * Ll1
        losses["simloss"] = self.config.ssim_lambda * simloss

        # sky acc loss
        accumulation = outputs["accumulation"]
        if gt_semantic is not None and self.config.sky_acc_loss_mult > 0:
            sky_mask = (gt_semantic == SemanticType.SKY.value).cuda()
            losses["sky_accumulation"] = self.config.sky_acc_loss_mult * (sky_mask * accumulation).mean()

        #NOTE: currently, adding reg loss does not work, as it makes training unstable
        # Object acc only gets computed late in training, after refinement stops
        # Until then (and when evaluating) this will not exist in the outputs and we don't include the term
        if "object_acc" in outputs:
            object_acc = torch.clamp(outputs["object_acc"].squeeze(-1), min=1e-5, max=1-1e-5)
            object_mask = outputs["obj_mask"]
            # Implementation based on official street gaussians code
            object_acc_entropy_loss = torch.where(
                object_mask,
                -(object_acc * torch.log(object_acc) + (1. - object_acc) * torch.log(1. - object_acc)),
                -torch.log(1. - object_acc)
            ).mean()

            # Add reg loss with some warming up strategy
            losses["object_acc_entropy_loss"] = self.config.object_acc_entropy_loss_mult *\
                  min(1.0, (self.step-self.config.background_model.stop_split_at) / 1000.0) * object_acc_entropy_loss
        else:
            losses['object_acc_entropy_loss'] = torch.tensor(0.0, device=rgb.device)

        return losses

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        metrics_dict = {}

        d = self._get_downscale_factor()
        if d > 1:
            newsize = (int(math.ceil(batch["image"].shape[0] / d)), int(math.ceil(batch["image"].shape[1] / d)))
            gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            gt_img = batch["image"]
            
        gt_rgb = gt_img.to(self.device)
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        metrics_dict["gaussian_count"] = self.num_points

        scales = outputs["scales"]
        metrics_dict["scale_mean"] = torch.exp(scales).mean()
        metrics_dict["log_scale_mean"] = scales.mean()

        opacities = outputs["opacities"]
        metrics_dict["sigmoid_opacity"] = torch.sigmoid(opacities).mean()

        return metrics_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.get_gt_img(batch["image"]).to(self.device)
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            predicted_rgb = TF.resize(outputs["rgb"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            predicted_rgb = outputs["rgb"]

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        if "mask" in batch:
            mask = batch["mask"].to(self.device).float()
            gt_rgb *= mask
            predicted_rgb *= mask
            
        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        #vis acc
        acc = colormaps.apply_colormap(outputs["accumulation"])
        combined_acc = torch.cat([acc], dim=1)
        images_dict = {"img": combined_rgb,"accumulation": combined_acc}

        # depth visualization
        if "depth" in batch:
            predicted_depth = outputs["depth"]
            depth_img = batch["depth"].to(self.device)
            if depth_img.dim() == 2:
                depth_img = depth_img.unsqueeze(-1)
            if predicted_depth.dim() == 2:
                predicted_depth = predicted_depth.unsqueeze(-1)
            # eval metric
            metrics_dict.update(self.depth_result(predicted_depth,depth_img))
            # vis
            predicted_depth = colormaps.apply_depth_colormap(
                predicted_depth,
                accumulation=outputs["accumulation"],
                )
            depth_img = colormaps.apply_depth_colormap(depth_img)
            combined_depth = torch.cat([depth_img, predicted_depth], dim=1)
            images_dict.update({"depth": combined_depth})
        else:
            predicted_depth = outputs["depth"]
            if predicted_depth.dim() == 2:
                predicted_depth = predicted_depth.unsqueeze(-1)
            predicted_depth = colormaps.apply_depth_colormap(
                predicted_depth,
                accumulation=outputs["accumulation"],
                )
            images_dict.update({"depth": predicted_depth})
        return metrics_dict, images_dict
    
    def load_state_dict(self, dict, **kwargs):  # type: ignore
        for model_name, model in self.all_models.items():
            sub_dict = {}
            for key in list(dict.keys()):
                if model_name in key:
                    sub_dict[".".join(key.split(".")[2:])] = dict.pop(key)
            model.load_state_dict(sub_dict, **kwargs)
        torch.nn.Module.load_state_dict(self, dict, strict=False)

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [image.shape[0] // d, image.shape[1] // d]
            return TF.resize(image.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        return image

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)
