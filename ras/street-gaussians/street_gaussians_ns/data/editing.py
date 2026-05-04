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
Tools for editing a scene's trajectories in various ways.

"""
from collections import OrderedDict
import math
from typing import Optional, Tuple
import torch
import copy
import yaml
from pathlib import Path
import bisect

from nerfstudio.utils.rich_utils import CONSOLE

from street_gaussians_ns.sgn_scene_graph import StreetGaussiansGraphModel
from street_gaussians_ns.data.utils.dynamic_annotation import Box, InterpolatedAnnotationSet, ObjectTrajectory


EDIT_TYPES = {
    # name: (required_attrs, optional_attrs)
    "accelerate_trajectory": (["object", "speed"], []),
    "swap_trajectories": (["object1", "object2"], []),
    "stop_vehicle": (["object"], ["time", "until_end"]),
    "reverse_trajectory": (["object"], []),
    "duplicate": (["object"], ["suffix", "offset"]),
    "remove": (["object"], []),
    "rescale": (["object", "scale"], []),
    "reverse": (["object"], []),
}


def load_edits(path: Path, model : StreetGaussiansGraphModel) -> list[dict]:
    """
    Load a specification of a set of scene edits from a YAML file.

    Verifies the data structure and returns a list of edit dicts that can be passed to `apply_edits()`.

    The scene model is required, so we can check the object IDs and so on.
    
    """
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if "edits" in data:
        edits, problems = preprocess_edits(data["edits"], model)
        if problems:
            raise IOError(
                "verification errors in YAML edits file: \n{}".format(
                    "\n".join(f" {p}: {prob}" for p, prob in enumerate(problems))
                )
            )
        else:
            return edits
    else:
        return []


def check_edit_attrs(edit : dict, model : StreetGaussiansGraphModel) -> Tuple[Optional[dict], list]:
    attrs = set(edit.keys())
    if "type" not in attrs:
        return None, ["type must be specified"]
    if edit["type"] not in EDIT_TYPES:
        return None, [f"unknown edit type '{edit['type']}'"]
    
    attrs.remove("type")

    problems = []
    required_attrs, optional_attrs = EDIT_TYPES[edit["type"]]

    # Check for missing required attrs for this edit type
    missing_attrs = set(required_attrs) - attrs
    for attr in missing_attrs:
        problems.append(f"required attribute '{attr}' missing")
    
    # Check for attrs not recognised at required or optional
    unknown_attrs = attrs - set(required_attrs) - set(optional_attrs)
    for attr in unknown_attrs:
        problems.append(f"unknown attribute '{attr}'")

    # Exclude the unknown attrs and try to carry on
    new_edit = dict((key, val) for (key, val) in edit.items() if key not in unknown_attrs)

    return new_edit, problems


def preprocess_edits(raw_data, model : StreetGaussiansGraphModel):
    edits = []
    problems = []

    if not isinstance(raw_data, list):
        problems.append("Edit data is not a list")
    else:
        for e_num, edit in enumerate(raw_data):
            # Check we have all the required attrs for the type
            edit, edit_problems = check_edit_attrs(edit, model)
            problems.extend([f"Edit {e_num}: {p}" for p in edit_problems])

            if edit is not None:
                edits.append(edit)
            
    return edits, problems


def check_objects(model : StreetGaussiansGraphModel, edit : dict):
    # Check any object references exist within the model
    # All (and only) object-reference attributes start with "object"
    missing_objects = []

    for attr, val in edit.items():
        if attr.startswith("object"):
            # This should refer to an object model in the scene graph
            if val not in model.object_annos.unique_track_ids:
                missing_objects.append(val)

    if missing_objects:
        raise ValueError(
            "could not resolve object reference in model: {}. Available objects: {}".format(
                ",".join(missing_objects),
                ", ".join(model.object_annos.unique_track_ids)
            )
        )


def apply_edits(model : StreetGaussiansGraphModel, edits : list[dict], start_time : int):
    """
    Apply scene edits (trajectory modification). Should be called once on a scene
    before rendering.

    """
    for edit in edits:
        edit_type = edit["type"]

        # Check objects exist
        # Do this now (not at the beginning), since other edits might have created new objects
        check_objects(model, edit)
        
        if edit_type == "accelerate_trajectory":
            CONSOLE.print(f"[bold green]Accelerating[/bold green] [blue]{edit['object']}[/blue] by [blue]{edit['speed']}[/blue]")
            accelerate_trajectory(model.object_annos, edit["object"], edit["speed"])
        elif edit_type == "swap_trajectories":
            CONSOLE.print(f"[bold green]Swapping trajectories[/bold green] of [blue]{edit['object1']}[/blue] and [blue]{edit['object2']}[/blue]")
            swap_trajectory(model.object_annos, edit["object1"], edit["object2"])
        elif edit_type == "stop_vehicle":
            # Time is given in seconds in edit config: convert to ms
            stop_time = start_time + edit["time"]*1000.
            until_end = edit.get("until_end", False)
            CONSOLE.print(f"[bold green]Stopping[/bold green] [blue]{edit['object']}[/blue] after time [blue]{stop_time}[/blue]")
            stop_trajectory_after(model.object_annos, edit["object"], stop_time, until_end=until_end)
        elif edit_type == "duplicate":
            # Allow a track ID suffix to be specified, or default to adding "_copy"
            suffix = edit.get("suffix", "_copy")
            offset = int(edit.get("offset", 1.) * 1000)

            CONSOLE.print(f"[bold green]Duplicating[/bold green] [blue]{edit['object']}[/blue] to make [blue]{edit['object'] + suffix}[/blue]")
            # Make a copy of the trajectory
            new_track_id = duplicate_trajectory(
                model.object_annos,
                edit["object"],
                suffix,
                time_offset=offset,
            )
            # Also copy the model itself
            duplicate_object_model(model, edit["object"], new_track_id)
        elif edit_type == "rescale":
            CONSOLE.print(f"[bold green]Rescaling[/bold green] [blue]{edit['object']}[/blue] by a factor of [blue]{edit['scale']}[/blue]")
            rescale_object(model, edit["object"], edit["scale"])
        elif edit_type == "remove":
            CONSOLE.print(f"[bold green]Removing[/bold green] [blue]{edit['object']}[/blue]")
            model.object_annos.remove_object_annotations(edit["object"])
        elif edit_type == "reverse":
            CONSOLE.print(f"[bold green]Reversing[/bold green] [blue]{edit['object']}[/blue]")
            reverse_trajectory(model.object_annos, edit["object"])
        else:
            raise ValueError(f"unknown edit type {edit_type}")


def accelerate_trajectory(annos: InterpolatedAnnotationSet, track_id: str, speed_factor: float):
    """
    Accelerates (>1.0) or decelerates (<1.0) the trajectory of a specific track_id.

    """
    if track_id not in annos.unique_track_ids:
        raise ValueError(f"Track {track_id} not found.")

    object_traj = annos.trajectories[track_id]
    original_traj = list(object_traj)
    if not len(original_traj):
        # Empty trajectory, nothing to do
        return

    start_time = object_traj.timestamps[0]
    end_time = object_traj.timestamps[-1]
    
    new_boxes = {}
    # Produce a new box for each of the timestamps in the original annotations until we reach
    #  either the end of the original trajectory or the end of the scene
    timestamps_to_scene_end = [t for t in annos.all_timestamps if t >= start_time]
    for timestamp in timestamps_to_scene_end:
        elapsed = float(timestamp - start_time)
        virtual_elapsed = elapsed * speed_factor
        virtual_time = start_time + virtual_elapsed
        
        if virtual_time > end_time:
            # Gone past the end of the trajectory
            break

        idx = bisect.bisect_right(object_traj.timestamps, virtual_time)
        if idx == 0:
            # Just use the old annotation for the start of the path
            new_box = original_traj[0][1]
        else: 
            # Get the last annotation before this virtual time
            t1, box1 = original_traj[idx-1]
            
            if virtual_time == t1:
                # We have an annotation for this exact time, so can just reuse that
                new_box = box1
            else:
                t2, box2 = original_traj[idx]
                interp_prop = (virtual_time - t1) / (t2 - t1)
                new_box = Box.interpolate(box1, box2, interp_prop)
        new_boxes[timestamp] = new_box

    # Add the new annotations, replacing the old ones
    annos.set_object_annotations(
        track_id,
        ObjectTrajectory(new_boxes)
    )


def swap_trajectory(anno_set: InterpolatedAnnotationSet, track_id_x: str, track_id_y: str):
    """
    Swaps the trajectory (path) of two objects. 
    Object X will appear where Object Y was, and vice versa.

    """
    if track_id_x not in anno_set.trajectories or track_id_y not in anno_set.trajectories:
        raise ValueError(f"One or both track IDs ({track_id_x}, {track_id_y}) not found")

    # Get static properties (size and label) for each trajectory
    box_x_sample = anno_set.trajectories[track_id_x].meta
    size_x, label_x = box_x_sample.size, box_x_sample.label
    box_y_sample = anno_set.trajectories[track_id_y].meta
    size_y, label_y = box_y_sample.size, box_y_sample.label

    # Mutate boxes: we'll simply swap the trajectories, so just change the sizes
    traj_x, traj_y = anno_set.trajectories[track_id_x], anno_set.trajectories[track_id_y]
    # Path X becomes path for Y
    for __, box in traj_x:
        box.size = size_y
        box.label = label_y
    # Path Y becomes path for X
    for __, box in traj_y:
        box.size = size_x
        box.label = label_x

    # Now use them for the swapped objects
    anno_set.set_object_annotations(track_id_x, traj_y)
    anno_set.set_object_annotations(track_id_y, traj_x)


def stop_trajectory_after(anno_set: InterpolatedAnnotationSet, track_id: str, stop_timestamp: int, until_end: bool = False):
    """
    Stops the trajectory of the object with track_id after the given timestamp.
    The object will remain at its last position from stop_timestamp onwards.

    stop_timestamp is given as an integer time in milliseconds.

    If until_end==True (default), the trajectory is extended to keep the object in
    its stopped position until the last time that any object has an annotation for.

    Otherwise, the trajectory only lasts as long as it did before, but the
    object is still for its duration

    """
    if track_id not in anno_set.trajectories:
        raise ValueError(f"Track {track_id} not found")
    
    # Find the last box before or at stop_timestamp
    obj_traj = anno_set.trajectories[track_id]
    if len(obj_traj) == 0:
        # Empty trajectory, nothing to do
        return

    # We only need to add a single point on the trajectory after this, so
    #  that this last position can be interpolated for any time after this
    if until_end:
        # Put the final (static) position at the same time as the last annotation on any object
        final_ts = anno_set.all_timestamps[-1]
    else:
        # Put the final position at what was previously the last annotation on this object
        final_ts = obj_traj.end_time

    # The box state at stop_timestamp
    stop_box = obj_traj.get_box_for_time(stop_timestamp, interpolate=True)
    if stop_box is None:
        return

    # ApearanceDuringEditing: this should be removed when we come up with a better solution for handling appearance changes during edits.
    # Freeze the appearance time to the stop timestamp
    stop_box.appearance_time = stop_timestamp

    # Remove all bbox annotations after the stop time
    keys_to_remove = [ts for ts in obj_traj.timestamps if ts > stop_timestamp]
    for ts in keys_to_remove:
        del obj_traj.bboxes[ts]

    # Add the stop box at the stop time
    obj_traj.bboxes[int(stop_timestamp)] = stop_box
    # Add the final position, which is effectively the same box state
    final_box = copy.deepcopy(stop_box)
    obj_traj.bboxes[final_ts] = final_box


def duplicate_trajectory(
    anno_set: InterpolatedAnnotationSet, 
    source_track_id: str, 
    suffix: str, 
    time_offset: int = 0, 
    position_offset: list[float] = [0.0, 0.0, 0.0]
):
    """
    Duplicates the trajectory of an existing object to create a new traffic agent.
    
    Args:
        source_track_id: The ID of the existing object to copy.
        suffix: Suffix to add to get the new track ID and object name
        time_offset: Shift in time (integer milliseconds). Positive values delay the new object.
        position_offset: Shift in position [x, y, z].
    """
    new_track_id = source_track_id + suffix

    if source_track_id not in anno_set.trajectories:
        raise ValueError(f"Source track '{source_track_id}' not found when trying to duplicate")
    if new_track_id in anno_set.trajectories:
        raise ValueError(f"Target track ID for duplication '{new_track_id}' already exists")
    
    # Copy the old object trajectory
    new_traj = copy.deepcopy(anno_set.trajectories[source_track_id])

    # ApearanceDuringEditing: this should be removed when we come up with a better solution for handling appearance changes during edits.
    # Preserve the original appearance time relative to the sequence
    # The box at t+offset should look like at t
    if time_offset > 0.:
        for original_ts, box in new_traj.bboxes.items():
            # If appearance_time is explicitly set (e.g. from previous edits), keep it
            # Otherwise, the appearance corresponds to the original timestamp
            if box.appearance_time is None:
                box.appearance_time = original_ts
            # Sicne we are using the original timestamp as the appearance time, we should not shift start_time_look and end_time_look, so that the appearance features are still looked up correctly based on the original timestamps
    
    if any(v != 0. for v in position_offset):
        # Add the position offset to all bboxes
        pos_offset_tensor = torch.tensor(position_offset)
        for __, box in new_traj:
            box.center += pos_offset_tensor.to(box.center.device)
    
    if time_offset > 0.:
        # Offset all the times of the annotations
        new_traj.bboxes = OrderedDict((ts+time_offset, box) for (ts, box) in new_traj.bboxes.items())

    anno_set.set_object_annotations(new_track_id, new_traj)

    return new_track_id


def duplicate_object_model(
    model: StreetGaussiansGraphModel, 
    source_track_id: str, 
    new_track_id: str, 
):
    # After creating a new trajectory, also copy original model to create a separate instance that follows new trajectory
    orig_object_name = model.get_object_model_name(source_track_id)
    new_object_name = model.get_object_model_name(new_track_id)
    new_model = copy.deepcopy(model.all_models[orig_object_name])
    model.all_models[new_object_name] = new_model


def rescale_object(model: StreetGaussiansGraphModel, track_id: str, scale: float):
    """
    Rescale an object.
    
    Args:
        source_track_id: The ID of the existing object to copy.
        suffix: Suffix to add to get the new track ID and object name
        time_offset: Shift in time (seconds). Positive values delay the new object.
        position_offset: Shift in position [x, y, z].
    """
    object_name = model.get_object_model_name(track_id)
    object_model = model.all_models[object_name]

    with torch.no_grad():
        # Scale the positions of the Gaussians
        object_model.means.data *= scale
        # Scale the actual size of the Ellipsoids (stored in log space)
        object_model.scales.data += math.log(scale)

    # Also rescale the bounding boxes associated with the trajectory
    traj = model.object_annos.trajectories[track_id]
    for __, box in traj:
        box.size *= scale


def reverse_trajectory(annos: InterpolatedAnnotationSet, track_id: str):
    if track_id not in annos.unique_track_ids:
        raise ValueError(f"Track {track_id} not found.")

    object_traj = annos.trajectories[track_id]
    original_bboxes = list(object_traj.bboxes.values())
    original_timestamps = list(object_traj.bboxes.keys())
    
    # The box appearance to should be also in reverse
    # The box at index 0 (start time) should look like the box at index -1 (end time)
    reversed_bboxes = []
    
    # Iterate backwards through boxes, but map them to forward timestamps
    for i, box in enumerate(reversed(original_bboxes)):
        new_box = copy.deepcopy(box)
        # ApearanceDuringEditing: this should be removed when we come up with a better solution for handling appearance changes during edits.
        # If the box has a recorded appearance_time, remap it to the new timestamp
        if new_box.appearance_time is not None:
            new_box.appearance_time = original_timestamps[-1 - i]
        reversed_bboxes.append(new_box)

    object_traj.bboxes = OrderedDict(zip(object_traj.timestamps, reversed_bboxes))
    # ApearanceDuringEditing: this should be removed when we come up with a better solution for handling appearance changes during edits.
    object_traj.start_time_look, object_traj.end_time_look = object_traj.end_time, object_traj.start_time
