# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Inference-only script for ORBIT-2.
Runs downscaling on all test inputs and saves predictions to ./results/.
No ground truth, no metrics, no comparison plots.

Usage (mirrors run_era5.sh):
    torchrun --nproc_per_node=1 /workspace/infer_orbit2.py \
        /workspace/checkpoints/global-finetune/global_9.5m_precipitation.yaml \
        --checkpoint /workspace/checkpoints/global-finetune/global_9.5m_precipitation.ckpt \
        --variable total_precipitation_24hr
"""

import climate_learn as cl
import torch
import os
import glob
import functools
from pathlib import Path
from argparse import ArgumentParser
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp import MixedPrecision
from datetime import timedelta
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from functools import partial
from typing import Optional, Dict, Any, Union, Callable, Iterable, List, Tuple

from climate_learn.models.hub.components.vit_blocks import Block
from torch.nn import Sequential
from climate_learn.models.hub.components.pos_embed import interpolate_pos_embed
from climate_learn.utils.fused_attn import FusedAttn
from climate_learn.utils.visualize import VisualizationConfig, TileProcessor, process_single_tile, stitch_tiles
from climate_learn.utils.loaders import load_architecture, get_data_variables, load_loss, load_transform
from climate_learn.metrics import MetricsMetaInfo
from climate_learn.data.processing.era5_constants import CONSTANTS
from utils import seed_everything, init_par_groups


# ---------------------------------------------------------------------------
# Model utilities  (identical to visualize_orbit2.py)
# ---------------------------------------------------------------------------

def validate_data_type(data_type):
    valid_types = ["bfloat16", "float32"]
    if data_type not in valid_types:
        raise ValueError(
            f"Invalid data_type '{data_type}'. Only {valid_types} are supported."
        )


def load_pretrained_weights(model, pretrain_path, device, tensor_par_size=1, tensor_par_group=None):
    world_rank = dist.get_rank()
    local_rank = 0

    if tensor_par_size > 1 and pretrain_path is not None:
        pretrain_path = pretrain_path + "_rank_" + str(world_rank)

    print("world_rank", world_rank, "pretrain_path", pretrain_path, flush=True)

    if world_rank < tensor_par_size:
        if pretrain_path is None:
            sys.exit("pretrain_path is None")
        elif os.path.exists(pretrain_path):
            _load_pretrained_weights(model, pretrain_path, device, world_rank)
        else:
            sys.exit("pretrain path does not exist")

    dist.barrier(device_ids=[local_rank])


def _load_pretrained_weights(model, pretrain_path, device, world_rank):
    checkpoint = torch.load(pretrain_path, map_location="cpu")
    print("Loading pre-trained checkpoint from: %s" % pretrain_path)
    pretrain_model = checkpoint["model_state_dict"]
    del checkpoint

    state_dict = model.state_dict()
    for k in list(pretrain_model.keys()):
        if k not in state_dict.keys():
            del pretrain_model[k]
        elif pretrain_model[k].shape != state_dict[k].shape:
            if k == "pos_embed":
                interpolate_pos_embed(model, pretrain_model, new_size=model.img_size)
            else:
                del pretrain_model[k]

    msg = model.load_state_dict(pretrain_model, strict=False)
    print(msg)
    del pretrain_model


def load_model_module(
    device,
    data_module,
    task,
    architecture=None,
    model=None,
    model_kwargs=None,
    test_loss=None,
    test_target_transform=None,
    **_ignored,
):
    lat, lon = data_module.get_lat_lon()
    if lat is None and lon is None:
        raise RuntimeError("Data module has not been set up yet.")

    if architecture:
        model = load_architecture(task, data_module, architecture, **model_kwargs)
    elif not isinstance(model, nn.Module):
        raise TypeError("'model' must be nn.Module")

    in_vars, out_vars = get_data_variables(data_module)

    test_losses, test_transforms = [], []
    for tl in test_loss:
        if isinstance(tl, str):
            from climate_learn.metrics import MetricsMetaInfo
            clim = None
            metainfo = MetricsMetaInfo(in_vars, out_vars, lat, lon, clim)
            test_losses.append(load_loss(device, model, tl, False, metainfo))
        elif callable(tl):
            test_losses.append(tl)

    for tt in (test_target_transform or []):
        if isinstance(tt, str):
            test_transforms.append(load_transform(tt, data_module))
        else:
            test_transforms.append(tt)

    return model, None, None, test_losses, None, None, test_transforms


# ---------------------------------------------------------------------------
# Inference loop — predictions only, no ground truth
# ---------------------------------------------------------------------------

def run_inference(
    mm,
    dm,
    dm_vis,
    out_list,
    out_transform,
    variable,
    src,
    device,
    div,
    overlap,
    input_files=None,
    n_ts_per_file=1,
    tensor_par_size=1,
    tensor_par_group=None,
):
    lat, lon = dm.get_lat_lon()
    out_channel = dm.out_vars.index(variable)
    in_channel  = dm.in_vars.index(variable)

    yout = len(lat)
    xout = len(lon)
    if dm.inp_root_dir == dm.out_root_dir:
        yout *= mm.superres_mag
        xout *= mm.superres_mag

    yinp = yout // mm.superres_mag
    xinp = xout // mm.superres_mag

    processor = TileProcessor(div, overlap, (yinp, xinp), (yout, xout), mm.superres_mag)
    os.makedirs("./results", exist_ok=True)

    for index, batch in enumerate(dm_vis.test_dataloader()):
        x, y = batch[:2]
        in_variables  = batch[2]
        out_variables = batch[3]

        for adj_index in range(x.shape[0]):
            tiles = []
            for vindex in range(div):
                for hindex in range(div):
                    coords = processor.get_tile_coordinates(hindex, vindex)
                    x_tile = x[:, :, coords.yi1:coords.yi2, coords.xi1:coords.xi2]
                    y_tile = y[:, :, coords.yo1:coords.yo2, coords.xo1:coords.xo2]
                    tile_result = process_single_tile(
                        mm, x_tile, y_tile,
                        in_variables, out_variables, out_list,
                        in_channel, out_channel,
                        out_transform, out_transform,
                        device, src, adj_index, coords, processor,
                    )
                    tiles.append(tile_result)

            images = stitch_tiles(tiles, processor, has_ground_truth=False)

            if dist.get_rank() == 0:
                pred = images["prediction"]
                if input_files:
                    file_idx = index // n_ts_per_file
                    ts_idx   = index  % n_ts_per_file
                    file_stem = Path(input_files[min(file_idx, len(input_files) - 1)]).stem
                    label = f"{file_stem}_{ts_idx}"
                else:
                    label = str(index)
                np.save(f"./results/{label}_preds.npy", pred)

                vmin, vmax = pred.min(), pred.max()
                plt.figure(figsize=(pred.shape[1] / 100, pred.shape[0] / 100))
                im = plt.imshow(pred, cmap="Blues", vmin=vmin, vmax=vmax)
                plt.colorbar(im)
                plt.savefig(f"./results/{label}_prediction.png")
                plt.close()

                print(f"[{label}] saved prediction shape={pred.shape} "
                      f"min={vmin:.4f} max={vmax:.4f}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description="ORBIT-2 inference — predictions only")
    parser.add_argument("config",        type=str, help="Path to YAML config")
    parser.add_argument("--checkpoint",  type=str, default=None)
    parser.add_argument("--variable",    type=str, default="total_precipitation_24hr")
    parser.add_argument("--master-port", type=str, default="29500")
    parser.add_argument("--data-type",   type=str, choices=["float32", "bfloat16"], default=None)
    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = str(os.environ["HOSTNAME"])
    os.environ["MASTER_PORT"] = args.master_port
    os.environ["WORLD_SIZE"]  = "1"
    os.environ["RANK"]        = "0"

    world_rank = 0
    local_rank = 0
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    dist.init_process_group("nccl", timeout=timedelta(seconds=7200000),
                            rank=world_rank, world_size=1)

    conf = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    pretrain_path       = args.checkpoint or conf["trainer"]["pretrain"]
    data_type           = args.data_type or "float32"
    validate_data_type(data_type)

    batch_size          = conf["trainer"]["batch_size"]
    buffer_size         = conf["trainer"]["buffer_size"]
    spatial_resolution  = conf["data"]["spatial_resolution"]
    preset              = conf["model"]["preset"]
    dict_out_variables  = conf["data"]["dict_out_variables"]
    dict_in_variables   = conf["data"]["dict_in_variables"]
    default_vars        = conf["data"]["default_vars"]
    low_res_dir         = conf["data"]["low_res_dir"]
    superres_mag        = conf["model"]["superres_mag"]

    try:
        do_tiling = conf["tiling"]["do_tiling"]
        div       = conf["tiling"]["div"]     if do_tiling else 1
        overlap   = conf["tiling"]["overlap"] if do_tiling else 0
    except Exception:
        div, overlap = 1, 0

    tensor_par_size  = conf["parallelism"]["tensor_par"]
    fsdp_size        = 1 // tensor_par_size
    simple_ddp_size  = 1
    seq_par_size     = 1

    # Use low_res_dir for both input and output — no ground truth needed
    data_key = next(k for k in dict_in_variables if k in low_res_dir)
    in_vars  = dict_in_variables[data_key]
    out_vars = dict_out_variables[data_key]

    _, data_par_group, tensor_par_group, _, fsdp_group, _ = init_par_groups(
        data_par_size=1,
        tensor_par_size=tensor_par_size,
        seq_par_size=seq_par_size,
        fsdp_size=fsdp_size,
        simple_ddp_size=simple_ddp_size,
        num_heads=conf["model"]["num_heads"],
    )

    model_kwargs = {
        "default_vars":     default_vars,
        "superres_mag":     superres_mag,
        "cnn_ratio":        conf["model"]["cnn_ratio"],
        "patch_size":       conf["model"]["patch_size"],
        "embed_dim":        conf["model"]["embed_dim"],
        "depth":            conf["model"]["depth"],
        "decoder_depth":    conf["model"]["decoder_depth"],
        "num_heads":        conf["model"]["num_heads"],
        "mlp_ratio":        conf["model"]["mlp_ratio"],
        "drop_path":        conf["model"]["drop_path"],
        "drop_rate":        conf["model"]["drop_rate"],
        "tensor_par_size":  tensor_par_size,
        "tensor_par_group": tensor_par_group,
        "FusedAttn_option": FusedAttn.DEFAULT,
    }

    # Both dirs point to low_res — stitch_tiles sees has_ground_truth=False
    dm = cl.data.IterDataModule(
        "downscaling",
        low_res_dir[data_key],
        low_res_dir[data_key],
        in_vars, out_vars=out_vars,
        data_par_size=1, data_par_group=data_par_group,
        subsample=1, batch_size=1, buffer_size=buffer_size, num_workers=0,
        div=div, overlap=overlap,
    ).to(device)
    dm.setup(stage="test")

    dm_vis = cl.data.IterDataModule(
        "downscaling",
        low_res_dir[data_key],
        low_res_dir[data_key],
        in_vars, out_vars=out_vars,
        data_par_size=1, data_par_group=data_par_group,
        subsample=1, batch_size=1, buffer_size=buffer_size, num_workers=0,
        div=1, overlap=0,
    ).to(device)
    dm_vis.setup(stage="test")

    load_fn = partial(
        load_model_module,
        task="downscaling",
        test_loss=["rmse", "pearson", "mean_bias"],
        test_target_transform=["denormalize", "denormalize", "denormalize"],
    )

    model, _, _, _, _, _, test_transforms = load_fn(
        device, data_module=dm, architecture=preset, model_kwargs=model_kwargs
    )

    denorm = test_transforms[0]
    model  = model.to(device)

    load_pretrained_weights(model, pretrain_path, device,
                            tensor_par_size=tensor_par_size,
                            tensor_par_group=tensor_par_group)

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Block, Sequential},
    )
    check_fn = lambda m: isinstance(m, Block) or isinstance(m, Sequential)

    precision_dt = torch.float32 if data_type == "float32" else torch.bfloat16
    bfloat_policy = MixedPrecision(
        param_dtype=precision_dt, reduce_dtype=precision_dt, buffer_dtype=precision_dt
    )

    model = FSDP(
        model, device_id=local_rank, process_group=fsdp_group,
        sync_module_states=True,
        sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=bfloat_policy,
        forward_prefetch=True, limit_all_gathers=False,
    )
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)

    in_shape, _ = dm.get_data_dims()
    _, _, in_height, in_width = in_shape
    with FSDP.summon_full_params(model):
        model.data_config(spatial_resolution[data_key], (in_height, in_width),
                          len(in_vars), len(out_vars))

    model.eval()
    seed_everything(0)

    input_files = sorted(glob.glob(os.path.join(low_res_dir[data_key], "test", "*")))
    if input_files:
        _d = np.load(input_files[0])
        _key = next(k for k in _d.files if _d[k].ndim >= 3)
        n_ts_per_file = _d[_key].shape[0]
    else:
        n_ts_per_file = 1

    with torch.no_grad():
        run_inference(
            model, dm, dm_vis,
            out_list=out_vars,
            out_transform=denorm,
            variable=args.variable,
            src=data_key,
            device=device,
            div=div, overlap=overlap,
            input_files=input_files,
            n_ts_per_file=n_ts_per_file,
            tensor_par_size=tensor_par_size,
            tensor_par_group=tensor_par_group,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
