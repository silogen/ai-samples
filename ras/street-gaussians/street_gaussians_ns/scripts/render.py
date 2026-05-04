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

import copy
import gc
import gzip
import json
import os
import re
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple, Union
from typing_extensions import Annotated

import mediapy as media
import numpy as np
import torch
import tyro

from rich import box, style
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.scripts.render import BaseRender
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs

from street_gaussians_ns.data.sgn_datamanager import FullImageDatamanagerConfig
from street_gaussians_ns.data.sgn_dataset import InputDataset
from street_gaussians_ns.sgn_scene_graph import StreetGaussiansGraphModel
from street_gaussians_ns.data.editing import load_edits, apply_edits
from street_gaussians_ns.data.cameras import Cameras
from street_gaussians_ns.data.utils.pytorch3d_functions import matrix_to_quaternion, quaternion_to_matrix, slerp


@contextmanager
def _disable_datamanager_setup(cls):
    """
    Disables setup_train or setup_eval for faster initialization.
    """
    old_setup_train = getattr(cls, "setup_train")
    old_setup_eval = getattr(cls, "setup_eval")
    old_cache_images = getattr(cls, "cache_images", None)
    setattr(cls, "setup_train", lambda *args, **kwargs: None)
    setattr(cls, "setup_eval", lambda *args, **kwargs: None)
    setattr(cls, "cache_images", lambda *args, **kwargs: (None,None))
    yield cls
    setattr(cls, "setup_train", old_setup_train)
    setattr(cls, "setup_eval", old_setup_eval)
    setattr(cls, "cache_images", old_cache_images)


@dataclass
class DatasetRender(BaseRender):
    """Render all images in the dataset."""

    output_path: Optional[Path] = None  # Path("renders")
    """Path to output video file."""
    data: Optional[Path] = None
    """Override path to the dataset."""
    edits: Optional[Path] = None
    """Path to a YAML file specifying edits to apply to the object trajectories before rendering"""
    downscale_factor: Optional[float] = None
    """Scaling factor to apply to the camera image resolution."""
    split: Literal["train", "val", "test", "train+test", "all"] = "all"
    """Split to render."""
    rendered_output_names: Optional[List[str]] = field(default_factory=lambda: None)
    """Name of the renderer outputs to use. rgb, depth, raw-depth, gt-rgb etc. By default all outputs are rendered."""
    output_format: Literal["images", "video", "images+video"] = "video"
    """How to save output data."""
    vehicle_config: Optional[Path] = None
    """Camera pose transform config on the new vehicle."""
    depth_near_plane: Optional[float] = 0.
    """Closest depth to consider when using the colormap for depth. If None, use min value."""
    depth_far_plane: Optional[float] = 3.
    """Furthest depth to consider when using the colormap for depth. If None, use max value."""
    load_sky_sphere_image_file: Optional[str] = None # "ldr_sky_sphere_inpainted_0.png"
    """Load sky sphere image"""
    export_sky_sphere_mask: bool = False
    """Export sky sphere sphere mask image, only support splatfacto model"""
    separate_submodels: bool = False
    """Output a separate video for each submodel in the scene graph"""
    model_names: List[str] = field(default_factory=lambda: ["full_scene", "all_objects", "background"])
    """Names of the models to render. Defaults to full_scene, all_objects, background."""
    fps: Optional[int] = None
    """
    Framerate of the output video. If this does not match the framerate of the input data,
    cameras and timestamps will be interpolated linearly to render each frame.

    If not given, no interpolation is performed and the output video has a framerate of 10fps.
    """

    def __post_init__(self):
        if self.output_path is None:
            self.output_path = self.load_config.parent/"renders"

    def main(self):
        config: TrainerConfig
        assert self.output_path is not None

        import torch
        if torch.__version__.split("+")[0] >= "2.1.0":
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True

        def update_config(config: TrainerConfig) -> TrainerConfig:
            data_manager_config = config.pipeline.datamanager
            assert isinstance(data_manager_config, (VanillaDataManagerConfig, FullImageDatamanagerConfig))
            data_manager_config.eval_num_images_to_sample_from = -1
            data_manager_config.eval_num_times_to_repeat_images = -1
            if isinstance(data_manager_config, VanillaDataManagerConfig):
                data_manager_config.train_num_images_to_sample_from = -1
                data_manager_config.train_num_times_to_repeat_images = -1
            if self.data is not None:
                data_manager_config.data = self.data
            if self.downscale_factor is not None:
                assert hasattr(data_manager_config.dataparser, "downscale_factor")
                setattr(data_manager_config.dataparser, "downscale_factor", self.downscale_factor)
            return config

        config, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
            update_config_callback=update_config,
        )
        data_manager_config = config.pipeline.datamanager
        assert isinstance(data_manager_config, (VanillaDataManagerConfig, FullImageDatamanagerConfig))
        # Don't cache on GPU when rendering
        data_manager_config.cache_images="cpu"

        model = pipeline.model
        # Turn off the bbox optimizer for rendering: otherwise it will transform the training bboxes,
        #  but any interpolated ones will not be transformed
        model.bbox_optimizer.config.mode
        assert isinstance(model, StreetGaussiansGraphModel), \
            f"this render script is only designed to render a StreetGaussiansGraphModel: loaded {type(model)}"
        
        CONSOLE.print(f"All submodel names: {', '.join(model.all_models.keys())}")

        # If scene edits have been specified, load and verify them
        if self.edits is not None:
            CONSOLE.rule("Editing", style="green")
            CONSOLE.print(f"Loading scene edits from {self.edits}")
            edits = load_edits(self.edits, model)

            # Load the cameras from the dataset to work out the start time
            with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                datamanager = data_manager_config.setup(test_mode="all", device="cpu")
            first_camera : Cameras = datamanager.eval_dataset.cameras[0]
            start_time = first_camera.time_ms
            CONSOLE.print(f"Start time: {start_time}")
 
            # Apply the edits to the scene model
            CONSOLE.print(f"Applying {len(edits)} edits to the scene")
            apply_edits(model, edits, start_time)
            CONSOLE.rule(style="green")

        # The full model (scene graph) renders all submodels together
        # Render all submodels, without the background
        models = self.model_names
        if self.separate_submodels:
            # Split up the submodels and generate from each separately
            models.extend(list(model.all_models.keys()))

        for model_name in models:
            base_output_path = self.output_path
            if len(models) > 1:
                # Distinguish the outputs for the separate submodels
                base_output_path = base_output_path / model_name

            for split in self.split.split("+"):
                datamanager: VanillaDataManager
                dataset: InputDataset
                # For rendering, don't cache any data on the GPU
                if split == "train":
                    with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                        datamanager = data_manager_config.setup(test_mode="test", device="cpu")

                    dataset = datamanager.train_dataset
                    dataparser_outputs = getattr(dataset, "_dataparser_outputs", datamanager.train_dataparser_outputs)
                else:
                    with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                        datamanager = data_manager_config.setup(test_mode=split, device="cpu")

                    dataset = datamanager.eval_dataset
                    dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
                    if dataparser_outputs is None:
                        dataparser_outputs = datamanager.dataparser.get_dataparser_outputs(split=datamanager.test_split)
                image_names_idx = [id for id, _ in sorted(enumerate(dataparser_outputs.image_filenames), key=lambda x: x[1],reverse=False)]
                if self.vehicle_config is not None:
                    self._transform_cameras_to_new_vehicle(dataset, dataparser_outputs)

                dataloader = FixedIndicesEvalDataloader(
                    input_dataset=dataset,
                    image_indices=image_names_idx,
                    device=datamanager.device,
                    num_workers=datamanager.world_size * 4,
                )

                # Check the framerate of the training data
                input_fps = get_input_framerate(dataloader, dataparser_outputs)

                # Generate a new set of poses for the trajectory that has the desired framerate
                # If this is not the same as the original framerate, we will interpolate between the original poses
                for camera_name, num_poses, pose_generator in interpolate_cameras(dataloader, dataparser_outputs, self.fps, input_fps):
                    with Progress(
                        TextColumn(f":movie_camera: Rendering {model_name}, split {split}, camera {camera_name} :movie_camera:"),
                        BarColumn(),
                        TaskProgressColumn(
                            text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
                            show_speed=True,
                        ),
                        ItersPerSecColumn(suffix="fps"),
                        TimeRemainingColumn(elapsed_when_finished=False, compact=False),
                        TimeElapsedColumn(),
                    ) as progress, ExitStack() as stack:
                        writers = {}
                        _writer_images = {}

                        for camera, batch, image_filename in progress.track(pose_generator, total=num_poses):
                            camera: Cameras = camera.to(model.device)

                            with torch.no_grad():
                                if model_name == "full_scene":
                                    outputs = model.get_outputs(camera)
                                elif model_name == "all_objects":
                                    outputs = model.get_outputs(camera, show_models=[m for m in model.all_models.keys() if m != "background"])
                                else:
                                    outputs = model.get_outputs(camera, show_models=[model_name])

                            for key in ["scales", "opacities"]:
                                outputs.pop(key, None)
                                
                            gt_batch = batch.copy()
                            gt_batch["rgb"] = gt_batch.pop("image")
                            all_outputs = (
                                list(outputs.keys())
                                + [f"raw-{x}" for x in outputs.keys()]
                                + [f"gt-{x}" for x in gt_batch.keys()]
                                + [f"raw-gt-{x}" for x in gt_batch.keys()]
                            )
                            rendered_output_names = self.rendered_output_names
                            if rendered_output_names is None:
                                if model_name != "full_scene":
                                    rendered_output_names = list(outputs.keys())
                                else:
                                    rendered_output_names = ["gt-rgb"] + list(outputs.keys())
                            for rendered_output_name in rendered_output_names:
                                if rendered_output_name not in all_outputs:
                                    CONSOLE.rule("Error", style="red")
                                    CONSOLE.print(
                                        f"Could not find {rendered_output_name} in the model outputs", justify="center"
                                    )
                                    CONSOLE.print(
                                        f"Please set --rendered-output-name to one of: {all_outputs}", justify="center"
                                    )
                                    sys.exit(1)
                                        
                                # Directory to output images to for this camera and rendered output
                                camera_output_path = base_output_path / split / rendered_output_name
                                if self.edits:
                                    # Add another subdir level if we applied edits, so that we keep the edited outputs separate from the unedited ones
                                    camera_output_path /= "edited"
                                camera_output_path = camera_output_path / camera_name

                                is_raw = False
                                is_depth = rendered_output_name.find("depth") != -1
                                is_semantic = rendered_output_name.find("semantic") != -1

                                output_path = camera_output_path / image_filename
                                output_name = rendered_output_name
                                if output_name.startswith("raw-"):
                                    output_name = output_name[4:]
                                    is_raw = True
                                    if output_name.startswith("gt-"):
                                        output_name = output_name[3:]
                                        output_image : torch.Tensor = gt_batch[output_name]
                                    else:
                                        output_image : torch.Tensor = outputs[output_name]
                                        if is_depth:
                                            # Divide by the dataparser scale factor
                                            output_image.div_(dataparser_outputs.dataparser_scale)
                                else:
                                    if output_name.startswith("gt-"):
                                        output_name = output_name[3:]
                                        output_image = gt_batch[output_name]
                                    else:
                                        output_image = outputs[output_name]
                                
                                # Map to color spaces / numpy
                                if is_raw:
                                    output_image = output_image.cpu().numpy()
                                elif is_depth:
                                    output_image = (
                                        colormaps.apply_depth_colormap(
                                            output_image,
                                            near_plane=self.depth_near_plane,
                                            far_plane=self.depth_far_plane,
                                            colormap_options=self.colormap_options,
                                        )
                                        .cpu()
                                        .numpy()
                                    )
                                elif is_semantic:
                                    if output_name.startswith("gt-"):
                                        output_image = (output_image.squeeze().cpu().numpy() * 100).astype(np.uint8)
                                    else:
                                        # Output image is logits
                                        output_image = (output_image.argmax(dim=-1).cpu().numpy() * 100).astype(np.uint8)
                                else:
                                    output_image = (
                                        colormaps.apply_colormap(
                                            image=output_image,
                                            colormap_options=self.colormap_options,
                                        )
                                        .cpu()
                                        .numpy()
                                    )
                                del output_name

                                # Save to file
                                if "video" in self.output_format.split("+"):
                                    render_width = int(output_image.shape[1])
                                    render_height = int(output_image.shape[0])
                                    output_filename = str(camera_output_path.with_suffix(".mp4"))

                                    if output_filename not in writers:
                                        # Initialize the writer now we know the image size
                                        camera_output_path.parent.mkdir(exist_ok=True, parents=True)
                                        video_fps = float(self.fps) if self.fps is not None else input_fps
                                        writers[output_filename] = stack.enter_context(
                                            media.VideoWriter(
                                                path=output_filename,
                                                shape=(render_height, render_width),
                                                fps=video_fps,
                                            )
                                        )
                                    else:
                                        # If we have images of different sizes from different cameras,
                                        #  we need to pad them to combine in one video
                                        video_height, video_width = writers[output_filename].shape
                                        if (video_height, video_width) != (render_height, render_width):
                                            if render_height > video_height or render_width > video_width:
                                                raise ValueError(f"cannot pad image of size ({render_height},{render_width}) to ({video_height},{video_width}): smaller shape came first")
                                            # Pad with 0s to make the right size
                                            height_pad = video_height-render_height
                                            top_pad = height_pad // 2
                                            bottom_pad = height_pad - top_pad
                                            width_pad = video_width-render_width
                                            left_pad = width_pad // 2
                                            right_pad = width_pad - left_pad
                                            output_image = np.pad(output_image, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)))
                                    _writer_images[output_filename] = _writer_images.get(output_filename, 0) + 1
                                    writers[output_filename].add_image(output_image)

                                if "images" in self.output_format.split("+"):
                                    output_path.parent.mkdir(exist_ok=True, parents=True)
                                    if is_raw:
                                        with gzip.open(output_path.with_suffix(".npy.gz"), "wb") as f:
                                            np.save(f, output_image)
                                    elif self.image_format == "png":
                                        media.write_image(output_path.with_suffix(".png"), output_image, fmt="png")
                                    elif self.image_format == "jpeg":
                                        media.write_image(
                                            output_path.with_suffix(".jpg"), output_image, fmt="jpeg", quality=self.jpeg_quality
                                        )
                                    else:
                                        raise ValueError(f"Unknown image format {self.image_format}")

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        for split in self.split.split("+"):
            table.add_row(f"Outputs {split}", str(self.output_path / split))
        CONSOLE.print(Panel(table, title=f"[bold][green]:tada: Render on split {split} Complete :tada:[/bold]", expand=False))


    def _transform_cameras_to_new_vehicle(self, dataset, dataparser_outputs):
        assert self.vehicle_config is not None
        dataparser_scale = dataparser_outputs.dataparser_scale
        new_vehicle_sensors = json.load(self.vehicle_config.open())

        cameras = dataset.cameras
        for camera_config in new_vehicle_sensors:
            image_path_patten = camera_config['image_path_patten']
            ca2cb = torch.tensor(camera_config['transform'], dtype=dataset.cameras.camera_to_worlds.dtype,
                                 device=dataset.cameras.camera_to_worlds.device)
            ca2cb[:3, 3] *= dataparser_scale

            p = re.compile(image_path_patten)
            for i, image_path in enumerate(dataparser_outputs.image_filenames):
                if not p.match(image_path.as_posix()):
                    continue

                ca2w = cameras.camera_to_worlds[i]
                row = torch.tensor([0,0,0,1], dtype=ca2w.dtype, device=ca2w.device).reshape((1,4))
                ca2w = torch.cat([ca2w, row])
                cb2w = torch.linalg.inv(ca2cb @ torch.linalg.inv(ca2w))
                cameras.camera_to_worlds[i] = cb2w[:3]

        dataset.cameras = cameras
        dataparser_outputs.cameras = cameras


def _interpolate_poses_for_camera(image_lst: list[Tuple[Cameras, dict, str]], fps: int) -> Iterable[Tuple[Cameras, dict, str]]:
    """
    Generator to produce all the interpolated poses for a given camera
    
    """
    cameras = [c for (c, __, __) in image_lst]
    camera_times = [c.time_ms for c in cameras]
    # Invert the camera2worlds to be world2cameras (viewmats), which we can interpolate
    bottom_row = torch.tensor(
        [0., 0., 0., 1.], 
        device=cameras[0].camera_to_worlds.device, 
        dtype=cameras[0].camera_to_worlds.dtype
    )
    viewmats = [
        torch.linalg.inv(
            torch.vstack((
                c.camera_to_worlds[0],
                bottom_row
            ))
        ) for c in cameras
    ]

    # All times are in ms
    camera_start_time = camera_times[0]
    total_camera_time = camera_times[-1] - camera_times[0]

    # To acheive the desired framerate we need to generate this many frame
    for frame_i in range(int(total_camera_time / 1000. * fps)):
        # Calculate the timepoint for this frame so we can interpolate cameras correctly
        frame_time = camera_start_time + frame_i / fps * 1000.
        # Find the cameras on either side of this timestamp
        try:
            next_cam_i = next(i for i, cam_time in enumerate(camera_times) if cam_time >= frame_time)
        except StopIteration:
            # Went beyond the end: presumably just a fp problem
            yield image_lst[-1]
            continue

        if next_cam_i == 0:
            # Still at the first training camera
            yield image_lst[0]
        else:
            previous_cam_i = next_cam_i - 1
            # How far between them are we?
            interp_prop = (frame_time - camera_times[previous_cam_i]) / (camera_times[next_cam_i] - camera_times[previous_cam_i])
            # If we're very close to one camera or another, don't interpolate
            if interp_prop < 0.01:
                yield image_lst[previous_cam_i]
            elif interp_prop > 0.99:
                yield image_lst[next_cam_i]
            else:
                # Interpolate the camera position
                interp_viewmat = _interpolate_poses(interp_prop, viewmats[previous_cam_i], viewmats[next_cam_i])
                # Invert to get the new cam2world
                interp_cam2world = torch.linalg.inv(interp_viewmat)
                # Put together a new camera
                name_base, __, ext = image_lst[previous_cam_i][2].partition(".")
                new_image_data = (
                    copy.deepcopy(image_lst[previous_cam_i][0]),  # camera
                    copy.deepcopy(image_lst[previous_cam_i][1]),  # batch data dict
                    f"{name_base}_{frame_i}.{ext}"                # Give the image a distinct imaginary name
                )
                # Update its camera pose to the interpolated position
                new_image_data[0].camera_to_worlds[0] = interp_cam2world[:3, :]
                # Also update its timestamp (in seconds), so objects get rendered at the interpolated position
                new_image_data[0].times[0] = frame_time / 1000.
                new_image_data[0].update_times_ms()

                yield new_image_data
                # Free up memory, as these Cameras get memory intensive for a lot of poses
                del new_image_data


def interpolate_cameras(dataloader: FixedIndicesEvalDataloader, dataparser_outputs: DataparserOutputs, target_fps: int|None, input_fps: float) \
        -> Iterable[Tuple[str, int, Iterable[Tuple[Cameras, dict, str]]]]:
    # Group input images (camera instances) by their camera name
    CONSOLE.print("Loading training poses and preparing interpolator")
    images_by_camera_name = collect_images_by_camera(dataloader, dataparser_outputs)

    # Check the framerate of the original input data to see if it (roughly) matches the requested framerate
    if target_fps is None or round(input_fps) == target_fps:
        # No need to interpolate: we can just use the original camera positions
        CONSOLE.print("Generating images at original poses, since training data matches target framerate")
        for camera_name, image_lst in images_by_camera_name.items():
            yield camera_name, len(image_lst), iter(image_lst)
    else:
        CONSOLE.print(f"Interpolating camera poses from original framerate of {input_fps} fps to {target_fps} fps")
        # Interpolate between these cameras to get the desired framerate
        for camera_name, image_lst in images_by_camera_name.items():
            # Work out how many frames we'll generate so we can show progress
            total_camera_time = (image_lst[-1][0].time_ms - image_lst[0][0].time_ms) / 1000.
            num_frames = int(total_camera_time * target_fps)

            yield camera_name, num_frames, _interpolate_poses_for_camera(image_lst, target_fps)
            # Free up memory before going on to the next camera
            del image_lst
            gc.collect()


def _interpolate_poses(prop, viewmat1, viewmat2):
    """ Linear interpolation between two view matrices (world2cameras) """
    interp_pos = (1.-prop)*viewmat1[:3, 3] + prop*viewmat2[:3, 3]

    # Convert the rotation matrices to quaternions so we can interpolate correctly
    quat1 = matrix_to_quaternion(viewmat1[:3, :3])
    quat2 = matrix_to_quaternion(viewmat2[:3, :3])
    # A negative dot product will mean we rotate the wrong way between the angles, taking the longer angular path
    if (quat1 * quat2).sum() < 0:
        # Negate quat: this is the same rotation, but ensures we take the shortest path when interpolating
        quat2 = -quat2
    interp_quat = slerp(quat1, quat2, prop)
    interp_rot = quaternion_to_matrix(interp_quat)

    new_viewmat = viewmat1.clone()
    new_viewmat[:3, :3] = interp_rot
    new_viewmat[:3, 3] = interp_pos
    return new_viewmat


def collect_images_by_camera(dataloader: FixedIndicesEvalDataloader, dataparser_outputs: DataparserOutputs) -> dict[str, list[Tuple[Cameras, dict, str]]]:
    images_root = Path(os.path.commonpath(dataparser_outputs.image_filenames))

    # Collect images by image_dir / camera name
    images_by_camera_name = {}

    for camera, batch in dataloader:
        camera_idx = batch["image_idx"]
        # Try to get the original filename
        image_name = dataparser_outputs.image_filenames[camera_idx].relative_to(images_root)
        if len(image_name.parts) > 1:
            # We have subdirectories within the images root
            # These will serve to split generated videos into separate cameras
            image_dir = image_name.parent
            image_filename = image_name.name
        else:
            # All images in the same dir: we need to split up the camera names from the filenames if possible
            if "_" in image_name.name:
                image_dir, __, image_filename = image_name.name.rpartition("_")
            else:
                # Couldn't split camera names from filenames: 
                #  just put everything in the same directory and hope for the best
                image_dir = "all_images"
                image_filename = image_name.name
        images_by_camera_name.setdefault(image_dir, []).append(
            (camera, batch, image_filename)
        )

    return images_by_camera_name


def get_input_framerate(dataloader: FixedIndicesEvalDataloader, dataparser_outputs: DataparserOutputs) -> float:
    """
    Calculate the input framerate based on the timings of the first and
    last poses of a single camera.
    
    """
    images_by_camera_name = collect_images_by_camera(dataloader, dataparser_outputs)

    # Check the framerate of the original input data to see if it (roughly) matches the requested framerate
    # To do this, we assume (without checking) that the timestamps are evenly spaced
    # Look just at the first camera, assuming the others have the same framerate
    first_camera = list(images_by_camera_name.keys())[0]
    first_camera_poses = [c for (c, __, __) in images_by_camera_name[first_camera]]
    first_camera_times = [c.time_ms for c in first_camera_poses]
    start_time = first_camera_times[0]
    end_time = first_camera_times[-1]
    # Timestamps are in milliseconds
    total_time = (end_time - start_time) / 1000.
    input_fps_f = len(first_camera_poses) / total_time

    return input_fps_f


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[DatasetRender, tyro.conf.subcommand(name="dataset")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa