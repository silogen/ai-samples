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

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import json
import os


def get_box_corners(center, dimensions, orientation):
    # Unpack dimensions, and quaternions
    length, width, height = dimensions
    q = orientation

    # Generate orthogonal bounding box vertex sets
    dx  = length / 2.0
    dy = width / 2.0
    dz = height / 2.0

    corners = np.array(
        [
            [dx, dy, dz],
            [-dx, dy, dz],
            [-dx, -dy, dz],
            [dx, -dy, dz],
            [dx, dy, -dz],
            [-dx, dy, -dz],
            [-dx, -dy, -dz],
            [dx, -dy, -dz],
        ]
    )

    # Use quaternions to create rotations and apply them to the vertex set
    rotation = R.from_quat([q[1], q[2], q[3], q[0]])  # Note: quaternion order is [x, y, z, w]
    rotated_corners = rotation.apply(corners)

    # Add local coordinates to center point coordinates to yield world coordinates
    world_corners = rotated_corners + center

    return world_corners


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="")

    parser.add_argument("--meta_file", type=str, default="transform.json")
    parser.add_argument("--annotation_file", type=str, default="annotation.json")
    parser.add_argument("--nuscenes", action="store_true")
    parser.add_argument("--seen_cameras", action="store_true")
    parser.add_argument("--draw_boxes", action="store_true")
    parser.add_argument("--track_moving", action="store_true")
    parser.add_argument("--box_padding", type=float, default=0.1, help="Ratio of box dimension to use for padding (default 0.1)")

    args = parser.parse_args()
    root_path = args.root_path + "/"
    transform_path = root_path + args.meta_file
    annotation_path = root_path + args.annotation_file

    ## nuscenes
    transform1 = np.array(
        [
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    # Read the transforms.json file
    with open(transform_path, "r") as f:
        transform_data = json.load(f)

    with open(annotation_path, "r") as f:
        annotation_data = json.load(f)

    frames = transform_data["frames"]
    annotation_frames = annotation_data["frames"]
    moving_gids = []
    for frame in tqdm(frames):
        # Get the file_path value
        file_path = frame["file_path"]
        c2w = np.array(frame["transform_matrix"])
        ## nerfstudio to opencv
        c2w[2, :] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[0:3, 1:3] *= -1
        if args.nuscenes:
            c2w = np.linalg.inv(transform1) @ c2w

        h = frame["h"]
        w = frame["w"]
        fl_x = frame["fl_x"]
        fl_y = frame["fl_y"]
        cx = frame["cx"]
        cy = frame["cy"]
        camera_matrix = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype="float32")
        intrinsic_matrix = np.array([[fl_x, 0, cx, 0], [0, fl_y, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        camera = frame["camera"]
        camera_model = frame["camera_model"]
        if camera_model == "OPENCV_FISHEYE":
            dist_coeffs = np.array([frame["k1"], frame["k2"], frame["k3"], frame["k4"]], dtype="float32")
        elif camera_model == "OPENCV":
            dist_coeffs = np.array([frame["k1"], frame["k2"], frame["p1"], frame["p2"]], dtype="float32")
        timestamp = frame["timestamp"]
        image_path = root_path + file_path
        mask_path = image_path.replace("images", "masks")
        mask_box_path = image_path.replace("images", "masks_box2d")
        mask_path = os.path.splitext(mask_path)[0]
        mask_path = mask_path + ".png"
        mask_box_path = os.path.splitext(mask_box_path)[0]
        mask_box_path = mask_box_path + ".png"
        directory = os.path.dirname(mask_path)
        # Check if the folder exists
        if not os.path.exists(directory):
            # If the folder does not exist, create it
            os.makedirs(directory)
        directory = os.path.dirname(mask_box_path)
        # Check if the folder exists
        if not os.path.exists(directory):
            # If the folder does not exist, create it
            os.makedirs(directory)
        # Set default image size and color (255 means white in grayscale)
        width, height = w, h
        color = 255

        # Create a new white grayscale image
        image_white = Image.new("L", (width, height), color)
        found_dict = next((d for d in annotation_frames if str(timestamp) in str(d.get("timestamp"))), None)

        # NOTE: this compute the axis-aligned 2D rectangle masks.
        # NOTE: it masks all moving objects, for example, predestrians
        if found_dict is not None:
            boxes = []

            for object in found_dict["objects"]:
                if args.seen_cameras and camera not in object["seen_cameras"]:
                    continue
                if object["is_moving"] or object["gid"] in moving_gids:
                    if args.track_moving:
                        moving_gids.append(object["gid"])
                    translation = object["translation"]
                    lwh = object["size"]
                    if args.nuscenes:
                        lwh[0], lwh[1] = lwh[1], lwh[0]
                    rotation = object["rotation"]
                    world_corners = get_box_corners(translation, lwh, rotation)
                    w2c = np.linalg.inv(c2w)
                    rvec = w2c[:3, :3]
                    tvec = w2c[:3, 3]
                    umin = w
                    vmin = h
                    umax = 0
                    vmax = 0
                    for m in world_corners:
                        m_1 = np.array([m[0], m[1], m[2], 1])
                        # Projective transformation, only convert to integer in the last step
                        points_3D = np.array([[m[0], m[1], m[2]]], dtype="float32").reshape(-1, 1, 3)
                        uv_homogeneous = intrinsic_matrix @ w2c @ m_1
                        if uv_homogeneous[2] > 0:
                            if camera_model == "OPENCV_FISHEYE":
                                points_2D, _ = cv2.fisheye.projectPoints(
                                    points_3D,
                                    cv2.Rodrigues(rvec)[0],
                                    np.ascontiguousarray(tvec),
                                    camera_matrix,
                                    dist_coeffs,
                                )
                                u = int(points_2D[0][0][0])
                                v = int(points_2D[0][0][1])
                            # elif(camera_model=="OPENCV"):
                            #     points_2D, _ = cv2.projectPoints(points_3D,cv2.Rodrigues(rvec)[0], np.ascontiguousarray(tvec), camera_matrix, dist_coeffs)
                            #     u = int(points_2D[0][0][0])
                            #     v = int(points_2D[0][0][1])
                            else:
                                u, v = (uv_homogeneous[:2] / uv_homogeneous[2]).astype(int)

                            umax = max(umax, u)
                            vmax = max(vmax, v)
                            umin = min(umin, u)
                            vmin = min(vmin, v)
                    if umin == w or vmin == h or umax == 0 or vmax == 0:
                        continue
                    umin = max (umin ,0)
                    vmin = max (vmin ,0)
                    umax = min (umax ,w - 1)
                    vmax = min (vmax ,h - 1)
                    
                    # Pad the bounding box by a ratio of its width/height (default 0.1)
                    padding_x = int((umax - umin) * args.box_padding)
                    padding_y = int((vmax - vmin) * args.box_padding)
                    box = [
                        max(umin - padding_x, 0),
                        max(vmin - padding_y, 0),
                        min(umax + padding_x, w - 1),
                        min(vmax + padding_y, h - 1),
                    ]
                    boxes.append(box)

            if len(boxes) > 0:
                if args.draw_boxes:
                    image = Image.open(image_path)
                    # Create ImageDraw object
                    draw = ImageDraw.Draw(image)
                    # Use ImageDraw to draw each box on the image
                    for box in boxes:
                        if 0 <= box[0] < w and 0 <= box[1] < h and 0 <= box[2] < w and 0 <= box[3] < h:
                            draw.rectangle(box, outline="blue", width=2)  # You can modify color and line width as needed
                    image.save(mask_box_path)

                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                A = np.ones((h, w)) * 255
                for box in boxes:
                    x_min, y_min, x_max, y_max = box
                    A[y_min:y_max, x_min:x_max] = 0

                # Convert the data type to uint8 to meet the image requirements of the Pillow library
                image_data = A.astype(np.uint8)
                image = Image.fromarray(image_data)
                image.save(mask_path)
            else:
                image_white.save(mask_path)
        else:
            image_white.save(mask_path)