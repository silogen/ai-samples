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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Type

import torch
from torch.nn import Parameter

from gsplat.cuda._torch_impl import _quat_to_rotmat

from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.rich_utils import CONSOLE

from street_gaussians_ns.utils import num_sh_bases, random_quat_tensor, RGB2SH, SH2RGB, idft, k_nearest_sklearn, save_rgb_image
from street_gaussians_ns.rendering import render_outputs, get_viewmat


@dataclass
class StreetGaussiansComponentModelConfig(ModelConfig):
    """
    Config for a 3DGS model that constitutes one component of a Street Gaussians scene graph.
    This is used to represent the background model of a scene, as well as each of the dynamic object models.

    The config includes parameters that are specific to each component. Parameters that
    apply to a whole scene are in the scene graph config (StreetGaussiansGraphModelConfig).

    """

    _target: Type = field(default_factory=lambda: StreetGaussiansComponentModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = False
    """If True, continue to cull gaussians post refinement.
    Original Street Gaussians code had True as the default, but due to a bug actually behaved as if it were False."""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0002
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    fourier_features_dim: int = 5
    """Dimension of the Fourier features used for the diffuse component of the SH."""
    fourier_features_scale: int = 1
    """Scale of the Fourier features used for the diffuse component of the SH."""


class StreetGaussiansComponentModel(Model):
    """
    3DGS model for a single sub-model in a scene graph, which may be the background model or one of
    the object models. A number of these (always including the background model) are combined into
    a scene graph for training and rendering.

    Note that this model cannot be trained on its own and is designed only to be used as a component
    in a scene graph. It can, however, be rendered on its own.

    """
    config: StreetGaussiansComponentModelConfig

    def __init__(self,*args,**kwargs,):
        if "seed_points" in kwargs:
            self.seed_points = kwargs["seed_points"]
        else:
            self.seed_points = None
        if "extent" in kwargs:
            self.extent = kwargs["extent"]
        self._model_idx_in_scene_graph: int = kwargs.get("model_idx_in_scene_graph", -1)

        self._refine_record_dict_accum = {}
        # This may get set if we're outputing debugging images during refinement
        # If it is set, we will output images, otherwise not
        self.debugging_cam = None
        # Track how many GSs get updated each iter for outputing stats
        self._num_updated_gs = []

        self.use_refinement_distance_filter = False

        super().__init__(*args, **kwargs)

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        if self.features_dc is not None:
            return self.features_dc[:, 0, :]
        return None

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @means.setter
    def means(self, value):
        setattr(self.gauss_params, "means" , value)

    @means.deleter
    def means(self):
        del self.gauss_params.means

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @scales.setter
    def scales(self, value):
        setattr(self.gauss_params, "scales" , value)

    @scales.deleter
    def scales(self):
        del self.gauss_params.scales

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @quats.setter
    def quats(self, value):
        setattr(self.gauss_params, "quats" , value)

    @quats.deleter
    def quats(self):
        del self.gauss_params.quats

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @features_dc.setter
    def features_dc(self, value):
        setattr(self.gauss_params, "features_dc" , value)

    @features_dc.deleter
    def features_dc(self):
        del self.gauss_params.features_dc

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @features_rest.setter
    def features_rest(self, value):
        setattr(self.gauss_params, "features_rest" , value)

    @features_rest.deleter
    def features_rest(self):
        del self.gauss_params.features_rest

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    @opacities.setter
    def opacities(self, value):
        setattr(self.gauss_params, "opacities" , value)

    @opacities.deleter
    def opacities(self):
        del self.gauss_params.opacities

    def populate_modules(self):
        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)

        distances, _ = k_nearest_sklearn(means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        dim_sh = num_sh_bases(self.config.sh_degree)
        
        if (
            self.seed_points is not None
            and not self.config.random_init
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.zeros(self.seed_points[1].shape[0], self.config.fourier_features_dim, 3)
            features_dc[:, 0, :3] = shs[:, 0, :3]
            # self.features_dc = torch.nn.Parameter(features_dc)
            features_rest = shs[:, 1:, :]
        else:
            features_dc = torch.zeros(num_points, self.config.fourier_features_dim, 3)
            features_dc[:, 0, :3] = torch.rand(num_points, 3)
            # self.features_dc = torch.nn.Parameter(features_dc)
            features_rest = torch.zeros((num_points, dim_sh - 1, 3))

        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        self.crop_box: Optional[OrientedBox] = None
        self.back_color = torch.zeros(3)
        self.step = 0
        # dict used for record refinement gs points number for wandb
        self.refine_record_dict={}
        
        self.xys_grad_norm = None
        self.radii = None
        self.vis_counts = None

    def get_fourier_features(self, prop: float):
        """
        Compute fourier features for a point at a given proportion of the object's trajectory by computing an inverse fourier transform
        
        """
        # Clip to (0, 1)
        prop = max(0., min(1., prop))
        normalized_frame = torch.tensor([prop]).to(self.device)
        t = normalized_frame * self.config.fourier_features_scale
        idft_base = idft(t, self.config.fourier_features_dim)[..., None]
        return torch.sum(self.features_dc*idft_base, dim=1, keepdim=True)

    def init_refinement_stats(self):
        """
        Reset the stats that are accrued for use in refinement.
        Called at init and after each refinement.

        """
        self.xys_grad_norm = torch.zeros(self.num_points, dtype=torch.float32, device=self.device)
        self.vis_counts = torch.zeros(self.num_points, dtype=torch.int32, device=self.device)
        self.max_2Dsize = torch.zeros(self.num_points, dtype=torch.float32, device=self.device)

    def update_refinement_stats(self, survivor_mask, xys_grad, radii, image_size):
        """
        Called by the scene graph to update the xys_grad on this separate sub-model.
        Updates the moving average of grad norms that is used for refinement.

           survivor_mask (N): model-local boolean mask of the gaussians that were visible in this update
           xys_grad (N, 3): grad of xys for the gaussians being updated
           radii (N): sizes of the gaussians being updated
        """
        if self.xys_grad_norm is None:
            self.init_refinement_stats()
        assert self.xys_grad_norm is not None
        assert self.vis_counts is not None
        assert self.max_2Dsize is not None

        max_image_size = float(max(image_size[0], image_size[1]))

        with torch.no_grad():
            # keep track of a moving average of grad norms
            grad_norm = xys_grad.norm(dim=-1)  # type: ignore
            self.vis_counts[survivor_mask] += 1
            # Divide every contribution to the grad norm by max_image_size, instead of doing it
            #  later during refinement: then we don't need to store the image size
            self.xys_grad_norm[survivor_mask] += (grad_norm[survivor_mask] / max_image_size)
            # update the max screen size, as a ratio of number of pixels
            self.max_2Dsize[survivor_mask] = torch.maximum(
                self.max_2Dsize[survivor_mask],
                radii[survivor_mask].to(torch.float32) / max_image_size,
            )
        
        self._num_updated_gs.append(survivor_mask.sum())

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][self._model_idx_in_scene_graph]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        optimizer.param_groups[0]["params"][self._model_idx_in_scene_graph] = new_params[0]
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """
        Duplicates a given set of gaussians and adds the parameters to the optimizer
        
        """
        param = optimizer.param_groups[0]["params"][self._model_idx_in_scene_graph]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"][self._model_idx_in_scene_graph] = new_params[0]

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        """
        Duplicates a given set of gaussians and adds the parameters to all optimizers
        
        """
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)
        
    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def output_debugging_image_for_params(self, output_file: Path, camera, override_colors=None):
        """
        Renders an image of just this sub-model. May be called during training to visualise
        what is happening to individual objects during training.
        
        :param output_file: File path to output the image to
        :param camera: Camera pose to use
        :param override_colors: List of (mask, color) pairs, where the gaussians specified by the mask
            will have their color overridden to be color. Used for highlight specific gaussians.
        
        """
        colors = torch.cat((self.features_dc, self.features_rest), dim=1)
        opacities = self.opacities

        if override_colors is not None:
            colors = colors.clone()
            opacities = opacities.clone()
            for idxs, color in override_colors:
                colors[idxs] = color
                opacities[idxs] = 1e10
        
        K, viewmat = get_viewmat(camera, 1, camera.camera_to_worlds)

        outputs, meta = render_outputs(
            camera.width,
            camera.height,
            K,
            viewmat,
            self.means,
            self.quats,
            self.scales,
            opacities,
            colors,
            sh_degree=self.config.sh_degree,
            output_names=["rgb"],
        )
        save_rgb_image(outputs["rgb"], filename=output_file)

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return

        self.refine_record_dict.clear()
        with torch.no_grad():
            # Convert reset_every from a number of refinements to a number of steps
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            # Only split/cull if we've seen every image since opacity reset (also allowing for the offset of refine_every)
            do_densification = (
                self.step < self.config.stop_split_at
                and self.step % reset_interval > self.num_train_data + self.config.refine_every
            )
            if do_densification:
                if self.xys_grad_norm is None:
                    return
                # then we densify
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                
                # Splitting
                # xys_grad_norm is already divided by the image size
                avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5
                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                self.refine_record_dict.update({"high_grads_count":high_grads.sum().item()})
                self.refine_record_dict.update({"avg_grad_norm":avg_grad_norm.nanmean().item()})
                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                self.refine_record_dict.update({"avg_max_2Dsize": self.max_2Dsize.nanmean().item()})
                self.refine_record_dict.update({"max_max_2Dsize": self.max_2Dsize.max().item()})
                self.refine_record_dict.update({"num_splits_before_grad_thresh": splits.sum().item()})
                splits &= high_grads
                self.refine_record_dict.update({"num_splits_after_grad_thresh": splits.sum().item()})
                nsamps = self.config.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)
                self.refine_record_dict.update({"refine_splits_count":splits.sum().item()})

                # Duplication
                dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                dups &= high_grads
                dup_params = self.dup_gaussians(dups)
                self.refine_record_dict.update({"refine_dups_count":dups.sum().item()})
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                    )

                if self.debugging_cam is not None:
                    # Output an image that highlights the gaussians that were split or dupped
                    green_color = torch.tensor([0., 1., 0.], dtype=torch.float32, device=self.device)
                    blue_color = torch.tensor([0., 0., 1.], dtype=torch.float32, device=self.device)
                    cam, output_dir = self.debugging_cam # type: ignore
                    self.output_debugging_image_for_params(
                        output_dir / f"debug_render_{self.step:04d}_1_split_dup.png", 
                        cam, 
                        [
                            (torch.where(splits), green_color),
                            (torch.where(dups), blue_color),
                        ]
                    )

                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )

                deleted_mask = self.cull_gaussians(splits_mask)
            elif self.step >= self.config.stop_split_at and self.config.continue_cull_post_densification:
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None

            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)

            # Offset all the opacity reset logic by refine_every so that we don't
            #  save checkpoints right when the opacity is reset
            if self.step < self.config.stop_split_at and self.step % reset_interval == self.config.refine_every:
                # Reset value is set to be twice of the cull_alpha_thresh
                reset_value = self.config.cull_alpha_thresh * 2.0
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
                )
                # reset the exp of optimizer
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][self._model_idx_in_scene_graph]
                param_state = optim.state[param]
                if "exp_avg" in param_state:
                    param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                    param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
        
        for key, val in self.refine_record_dict.items():
            self._refine_record_dict_accum[key] = self._refine_record_dict_accum.get(key, 0) + val

        if self.step < self.config.stop_split_at:
            # Reset the stats to start accumulating afresh
            self.init_refinement_stats()

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        # cull transparent ones
        low_alpha = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        culls = low_alpha
        self.refine_record_dict.update({"refine_culls_alpha_count":low_alpha.sum().item()})
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
                # self.refine_record_dict.update({"refine_culls_toobigs_count":torch.sum(toobigs).item()})
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
            self.refine_record_dict.update({"refine_culls_toobigs_count":toobigs_count})

            if self.debugging_cam is not None:
                # Output an image that highlights the gaussians that are being culled
                # (not extra_cull_mask, which is those that were split)
                highlights = torch.where(low_alpha | toobigs)
                red_color = torch.tensor([1., 0., 0.], dtype=torch.float32, device=self.device)
                cam, output_dir = self.debugging_cam   # type:ignore
                self.output_debugging_image_for_params(
                    output_dir / f"debug_render_{self.step:04d}_2_cull.png", 
                    cam, 
                    [(highlights, red_color)]
                )

        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        # step 1, sample new means
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = _quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        # n_dups = dup_mask.sum().item()
        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]
        return new_dups
    
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        # Migration: removed after_train, as we no longer need to do anything there
        # The stats needed for object models' refinement get updated by the graph model's after_train

        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    
