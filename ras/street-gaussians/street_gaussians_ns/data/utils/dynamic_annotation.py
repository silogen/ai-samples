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

from collections import OrderedDict
import json
from operator import itemgetter
from typing import Iterable, Iterator, Optional, Tuple
import warnings
from scipy.spatial.transform import Rotation as R
import numpy as np
import open3d as o3d
from pathlib import Path
import bisect

import argparse
import torch
from functools import reduce

from street_gaussians_ns.data.utils.pytorch3d_functions import matrix_to_quaternion, quaternion_to_matrix, slerp


# only keep the box with label in FILTER_LABEL, if is None, select all
# FILTER_LABEL = None
FILTER_LABEL = ['car']


class Box:
    """
    3D bounding box of a dynamic object.

    """
    def __init__(
        self,
        center: torch.Tensor,
        size : torch.Tensor,
        rot : Optional[torch.Tensor] = None,
        quat : Optional[torch.Tensor] = None,
        yaw : Optional[float] = None,
        label : Optional[str] = None,
        interpolated: bool = False,
        appearance_time: Optional[int] = None,
    ) -> None:
        self.device = None
        self.center = center
        self.size = size
        # A string label may be set on the box (though we don't currently use this for anything)
        self.label = label
        # Indicates whether the box is produced by interpolation or an original annotation
        self.interpolated = interpolated

        # ApearanceDuringEditing: this should be removed when we come up with a better solution for handling appearance changes during edits.
        # Overridden time to use for looking up appearance (Fourier features)
        # If None, use the actual timestamp of the box
        self.appearance_time = appearance_time

        # We can set either rot or quat and the other will get computed from it
        if rot is not None:
            assert rot.shape == (3, 3)
            self.rot = rot
        elif quat is not None:
            self.quat = quat
        elif yaw is not None:
            # Yaw was given, compute a rot matrix from it and set rot and quat
            self.rot = torch.tensor(R.from_euler('xyz', [0, yaw, 0]).as_matrix())
        else:
            raise ValueError("one of rot, quat or yaw must be given when creating a Box")
    
    @property
    def rot(self):
        """3D rotation matrix, equivalent rotation to self.quat"""
        if self._rot is None:
            self._rot = quaternion_to_matrix(self._quat)
        return self._rot
    
    @rot.setter
    def rot(self, r):
        self._quat = matrix_to_quaternion(r)
        self._rot = r

    @property
    def quat(self):
        """3D rotation quaternion, equivalent rotation to self.rot"""
        return self._quat
    
    @quat.setter
    def quat(self, q):
        self._quat = q
        self._rot = None

    def to(self, device):
        self.center = self.center.to(device)
        self._quat = self._quat.to(device)
        if self._rot is not None:
            self._rot = self._rot.to(device)
        self.device = device
        return self

    def get_inliers_outliers(self, pcd):
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(self.vertices)
        )
        inliers_indices = obb.get_point_indices_within_bounding_box(pcd.points)
        # select inside points = cropped
        inliers_pcd = pcd.select_by_index(inliers_indices, invert=False)
        # select outside points
        outliers_pcd = pcd.select_by_index(
            inliers_indices, invert=True
        )
        return inliers_pcd, outliers_pcd

    @property
    def vertices(self):
        if self.size is None:
            raise ValueError("cannot get world-space vertices of box with no size")
        # x, y and z coordinates of all of the cuboid vertices in object space
        vertices = torch.tensor([
            [-1, -1, -1, -1, 1, 1, 1, 1],
            [-1, 1, 1, -1, -1, 1, 1, -1],
            [-1, -1, 1, 1, -1, -1, 1, 1]
        ]).T * self.size / 2.
        vertices = vertices @ self.rot.T
        vertices = vertices + self.center
        return vertices

    def to_mesh(self, c=[0.9, 0.1, 0.1]):
        assert self.size is not None
        w, h, length = self.size
        x, y, z = self.center
        mesh_box = o3d.geometry.TriangleMesh.create_box(
            width=w, height=h, depth=length)  # x y z
        # set anchor to center
        mesh_box.compute_vertex_normals()
        mesh_box.paint_uniform_color(c)
        mesh_box.rotate(self.rot.cpu().detach().numpy())
        mesh_box.translate([x, y, z])
        mesh_box.translate([-w/2, -h/2, -length/2], relative=True)
        return mesh_box

    def transform(self, translation : torch.Tensor, rotation : torch.Tensor):
        self.center = torch.matmul(rotation, self.center)+translation
        self.rot = torch.matmul(rotation, self.rot)

    def scale(self, scale_factor: float):
        """Applies a scaling factor to the whole scene, i.e. scales both the size and the position"""
        self.center *= scale_factor
        self.size *= scale_factor

    @staticmethod
    def interpolate(box1, box2, proportion):
        """Interpolate between two boxes to produce a new box"""
        i_center = box1.center.detach()*(1-proportion) + box2.center.detach()*proportion
        quat1 = box1.quat.detach()
        quat2 = box2.quat.detach()
        # A negative dot product will mean we rotate the wrong way between the angles, taking the longer angular path
        if (quat1 * quat2).sum() < 0:
            # Negate quat: this is the same rotation, but ensures we take the shortest path when interpolating
            quat2 = -quat2
        i_quat = slerp(quat1, quat2, proportion)

        # ApearanceDuringEditing: this should be removed when we come up with a better solution for handling appearance changes during edits.
        # Interpolate appearance time if present
        i_app_time = None
        if box1.appearance_time is not None and box2.appearance_time is not None:
             i_app_time = box1.appearance_time * (1-proportion) + box2.appearance_time * proportion
        elif box1.appearance_time is not None:
            # If transitioning from fixed time to normal (unlikely for stop) or vice versa
            i_app_time = box1.appearance_time
        elif box2.appearance_time is not None:
            i_app_time = box2.appearance_time

        box = Box(
            i_center,
            box1.size,
            quat=i_quat,
            label=box1.label,
            interpolated=True,
            appearance_time=i_app_time
        )
        return box


class ObjectTrajectory:
    """
    A set of annotations for a single dynamic object.
    
    """
    def __init__(
            self,
            bboxes : dict[int, Box],
            seed_pts : Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        # Mapping integer millisecond timestamp to Box object representing the bbox at that time
        # Make sure they're sorted by timestamp
        self.bboxes = OrderedDict(sorted(bboxes.items(), key=itemgetter(0)))
        # Seed points for this object
        self.seed_pts = seed_pts

        # ApearanceDuringEditing: this should be removed when we come up with a better solution for handling appearance changes during edits.
        # We keep track of the original start and end time of the trajectory, but we may want to look up appearance features using a different time range (e.g. if we reversed the trajectory)
        self.start_time_look = self.start_time
        self.end_time_look = self.end_time
    
    @property
    def start_time(self) -> int:
        return self.timestamps[0]
    
    @property
    def end_time(self) -> int:
        return self.timestamps[-1]
    
    @property
    def meta(self):
        """Just use the first annotation box as meta"""
        return next(iter(self.bboxes.values()))
    
    @property
    def timestamps(self):
        return list(self.bboxes.keys())
    
    def get_trajectory(self) -> list[Tuple[int, Box]]:
        """Get the object's annotations as a list of (timestamp, Box) pairs"""
        return list(self.bboxes.items())
    
    def __iter__(self) -> Iterator[Tuple[int, Box]]:
        return iter(self.bboxes.items())
    
    def __len__(self):
        return len(self.bboxes)
    
    def __getitem__(self, i) -> Tuple[int, Box]:
        ts = list(self.bboxes.keys())[i]
        return ts, self.bboxes[ts]
    
    def get_box_for_time(self, timestamp: int, interpolate: bool = False) -> Box|None:
        """
        Get a Box annotation for the given timestamp.
        
        If the timestamp falls outside the range of times on this object's trajectory, returns None.

        If an exact annotation for this timestamp is found, it is returned as is.

        Otherwise, if interpolate==False, returns None. If interpolate==True, interpolates to get
        a Box for this time.

        """
        if timestamp in self.bboxes:
            # We have an annotation for this exact timestamp, so don't interpolate
            return self.bboxes[timestamp]
        if not interpolate:
            # No annotation for this time and we've been asked not to interpolate
            return None
        if not self.start_time < timestamp < self.end_time:
            # Outside the time range of this object's trajectory: cannot interpolate
            return None
        
        # Between two existing frames, interpolate
        nearest_frame_id = bisect.bisect(self.timestamps, timestamp)
        assert nearest_frame_id > 0, "cannot intersect before the first timestamp"

        box1, box2 = self.bboxes[self.timestamps[nearest_frame_id-1]], self.bboxes[self.timestamps[nearest_frame_id]]
        t = (timestamp - self.timestamps[nearest_frame_id-1]) / (self.timestamps[nearest_frame_id] - self.timestamps[nearest_frame_id-1])
        return Box.interpolate(box1, box2, t)


def load_object_3D_points(ply_path: Path, scale_factor: float):
    if not ply_path.exists():
        return None
    # assert ply_path.exists(), f"{ply_path} not exists"
    pcd = o3d.io.read_point_cloud(str(ply_path))
    # read points_xyz
    points3D = torch.from_numpy(np.array(pcd.points, dtype=np.float32))
    if points3D.shape[0] < 10000:
        return None
    points3D *= scale_factor
    # Load point colours
    if pcd.has_colors():
        points3D_rgb = torch.from_numpy(np.array(pcd.colors, dtype=np.float32)).float() * 255.
    else:
        points3D_rgb = torch.rand(points3D.shape[0], 3, dtype=torch.float32) * 255.
        
    return (points3D, points3D_rgb)


class InterpolatedAnnotationSet:
    def __init__(
            self, 
            trajectories : dict[str, ObjectTrajectory] = {},
    ) -> None:
        # Annotation set for each dynamic object (indexed by trackID)
        self.trajectories = trajectories
        # List of all timestamps from all objects' annotations
        self._update_timestamps()

    def _update_timestamps(self):
        self.all_timestamps: list[int] = list(sorted(set(sum((ann.timestamps for ann in self.trajectories.values()), []))))
    
    @property
    def unique_track_ids(self) -> list[str]:
        """List of track IDs for all the objects we have annotations for"""
        return list(self.trajectories.keys())
    
    def timestamp_to_frame_num(self, timestamp):
        """
        Returns the position of the given timestamp in the list of all unique
        timestamps in the annotations, i.e. the frame number in the sequence.
        
        """
        return self.all_timestamps.index(timestamp)

    def __len__(self):
        return len(self.all_timestamps)

    def get_boxes_for_time(self, timestamp: int, interpolate: bool = False) -> list[Tuple[str, Box]]:
        """
        Collect boxes from all objects for the given timestamp.

        If interpolate=False, only boxes with the exact timestamp will be returned.
        This should be used at training time, where we expect to have annotations for
        every timestamp.

        If interpolate=True, boxes will be produced by interpolation for every object
        where the timestamp falls within the time range for which it is visible.
        This should be used when rendering a trained scene, where we may want to render
        frames that were not originally annotated.
        
        """
        boxes = []
        for obj_id, anno in self.trajectories.items():
            obj_box = anno.get_box_for_time(timestamp, interpolate=interpolate)
            if obj_box is not None:
                boxes.append((obj_id, obj_box))
        return boxes

    def iter_boxes(self) -> Iterable[Tuple[int, Box]]:
        """Iterate over all boxes (annotations) from all objects"""
        for obj_annos in self.trajectories.values():
            for timestamp, box in obj_annos.bboxes.items():
                yield timestamp, box

    def to_mesh(self):
        return merge_mesh(list(box for (t, box) in self.iter_boxes()))

    def get_trajectory(self, track_id: str):
        """
        Returns a sorted list of (timestamp_str, box) tuples for the given track_id.

        """
        return self.trajectories[track_id].get_trajectory()

    def remove_object_annotations(self, track_id: str):
        """
        Completely removes a track from annotations.
        """
        del self.trajectories[track_id]
        self._update_timestamps()

    def set_object_annotations(self, track_id: str, object_annos: ObjectTrajectory):
        """
        Adds a set of annotations for a given track_id (object). This will
        replace any existing annotations for this object. If you want to add
        new annotations on top of existing ones, edit the existing object
        annotations via `annotation_set.annos[track_id]`.
        
        :param track_id: Object ID
        :type track_id: str
        :param object_annos: New set of annotations
        :type object_annos: ObjectTrajectory

        """
        if track_id in self.trajectories:
            del self.trajectories[track_id]
        self.trajectories[track_id] = object_annos
        self._update_timestamps()

    @staticmethod
    def load(
            anno_json_path: Path, 
            lidar_path: Path, 
            self_car_label: Optional[str] = None, 
            scale_factor: float = 1., 
            ignore_static: bool = True,
            transform_matrix: Optional[torch.Tensor] = None,
            expand_bboxes: Optional[Tuple[float, float, float]] = None
    ):
        assert anno_json_path.exists()

        if expand_bboxes is None:
            expand_bboxes = torch.tensor([1., 1., 1.])
        else:
            # All bboxes will be slightly expanded
            expand_bboxes = torch.tensor(expand_bboxes)

        # Load JSON annotations
        with open(anno_json_path, "r") as f:
            annos = json.load(f)["frames"]
        annos = sorted(annos, key=lambda x: x['timestamp'])

        # Pre-compute moving objects: identify any object that moves at least once
        moving_track_ids = set()
        for frame in annos:
            for obj in frame.get('objects', []):
                if obj.get('is_moving', False):
                    moving_track_ids.add(obj['gid'])
        
        object_annos = dict((oid, OrderedDict()) for oid in moving_track_ids)
        for item in annos:
            # Timestamps are given in seconds as a float
            # We convert to millisecond integers and always use them in that form afterwards
            timestamp = int(item["timestamp"]*1000)
            
            boxes = []
            for obj in item["objects"]:
                if FILTER_LABEL is not None:
                    if obj['type'] not in FILTER_LABEL and not obj['type'].endswith('Car'):
                        continue
                
                # Check if the object EVER moves in the scene
                # Only ignore if it is completely static throughout the sequence
                if ignore_static and obj['gid'] not in moving_track_ids:
                    continue

                if self_car_label is not None:
                    # Skip the self car if an ID has been given for it
                    if obj['gid'] == self_car_label:
                        continue
                center = torch.tensor(obj['translation'])
                quat = torch.tensor(obj['rotation'])
                trackId = obj['gid']
                size = expand_bboxes*np.array(obj['size'])
                box = Box(
                    center, 
                    size,
                    label=obj['type'],
                    quat=quat
                )
                if transform_matrix is not None:
                    box.transform(transform_matrix[:3, 3], transform_matrix[:3, :3])
                if scale_factor != 1.:
                    box.scale(scale_factor)
                boxes.append(box)

                object_annos[trackId][timestamp] = box
            
        object_annotations = {}
        for trackId in moving_track_ids:
            # Load PLY points to init each object
            ply_path = lidar_path / f"{trackId}.ply"
            if not ply_path.exists():
                continue
            pts = load_object_3D_points(ply_path, scale_factor)
            if pts is None:
                warnings.warn(f"Could not load PLY initial points from {ply_path} for object {trackId}")
                seed_pts = None
            else:
                seed_pts = pts

            object_annotations[trackId] = ObjectTrajectory(object_annos[trackId], seed_pts)
        return InterpolatedAnnotationSet(object_annotations)


def merge_mesh(meshes):
    return reduce(lambda x, y: x+y, meshes)


def upper_bound(nums, target):
    low, high = 0, len(nums)-1
    pos = len(nums)
    while low < high:
        mid = (low+high)//2
        if nums[mid] <= target:
            low = mid+1
        else:  # >
            high = mid
            pos = high
    if nums[low] > target:
        pos = low
    return pos


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='prepapre annos.pkl for vdbfusion')
    parser.add_argument('--pcds_path', type=str, help='dataset path to pcd files',
                        default=None,
                        )
    parser.add_argument('--annos_path', type=str, help='dataset path to pcd files',
                        default=''
                        )
    parser.add_argument('--dataset_type', type=str, default='WaymoDataset',
                        choices=['WaymoDataset'])
    parser.add_argument('--anno_dst', type=str, default='./output/annos.pkl')
    parser.add_argument('--transform_json', type=str,
                        help='dataset path to pcd files', default=None)

    args = parser.parse_args()
    
    if not (Path(args.annos_path)/"annotation.json").exists():
        exit()
    annos = InterpolatedAnnotationSet.load(
        anno_json_path=Path(args.annos_path) / "annotation.json",
        lidar_path=Path(args.annos_path) / "aggregate_lidar" / "dynamic_objects"
    )
    dst=Path('/home/qinglin.yang/qinglin/debug_anno')
    dst.mkdir(parents=True,exist_ok=True)
        
    COLORMAPS = [
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0],  # Cyan
        [0.5, 0.5, 0.5],  # Gray
        [1.0, 0.5, 0.0],  # Orange
        [0.5, 0.0, 1.0],  # Purple
        [0.0, 0.5, 0.5]   # Teal
    ]

    for i, (trackId, obj_annos) in enumerate(annos.trajectories.items()):
        for j, (t, anno) in enumerate(obj_annos.bboxes.items()):
            if i%10:
                continue
            mesh = []
            c=COLORMAPS[annos.unique_track_ids.index(trackId)%len(COLORMAPS)]
            mesh.append(anno.to_mesh(c=c))
            o3d.io.write_triangle_mesh((dst/f'anno_mesh_{t}.ply').as_posix(),merge_mesh(mesh))
