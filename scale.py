import argparse
import os
from glob import glob

import numpy as np
import plotly.graph_objects as go
import pyransac3d
from skimage import io
from tqdm.auto import tqdm


def generate_pointcloud_by_depth(depth, intrinsics, colors=None, mask=None, color=(0, 255, 0)):
    """Generate pointcloud by depth and intrinsics

    Parameters
    ----------
    depth : ndarray (height, width)
        Depth predicted by SfM Learner
    intrinsics : [fx, fy, cx, cy]
        Inner camera parameters
    colors : ndarray (height, width)
        RGB image
    mask : ndarray (height, width)
        Road segmentation mask
    color : (r, g, b)
        Color to fill pointcloud if colors is not set

    Returns
    -------
    pointcloud : ndarray (N, 6)
        point cloud in x, y, z, r, g, b format
    """
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)

    fx, fy, cx, cy = intrinsics

    z = depth
    x = z * (c - cx) / fx
    y = z * (r - cy) / fy

    points = np.stack([x, y, z], axis=-1)

    if colors is None:
        colors = color * np.ones_like(points)
    if mask is None:
        mask = np.ones_like(depth, dtype=int)

    pointcloud = np.dstack([points, colors])

    return pointcloud[(z > 0) & (mask > 0)]


def generate_pointcloud_by_equation(eq, color=(0, 255, 0), bbox=(-10, 0, 20, 20)):
    """Generate pointcloud by plane equation

    Parameters
    ----------
    eq : [A, B, C, D] for Ax + By + Cz + D = 0 equation
        Plane equation
    color : (r, g, b)
        Color to fill plane
    bbox : [x, y, h, w]
        where to draw plane

    Returns
    -------
    pointcloud : ndarray (N, 6)
        point cloud in x, y, z, r, g, b format
    """
    c, r = np.meshgrid(np.linspace(bbox[0], bbox[0] + bbox[2], 100),
                       np.linspace(bbox[1], bbox[1] + bbox[3], 100))

    if abs(eq[1]) > 0:
        x = c.ravel()
        z = r.ravel()
        y = -(eq[0] * x + eq[2] * z + eq[3]) / eq[1]
    elif abs(eq[0]) > 0:
        y = c.ravel()
        z = r.ravel()
        x = -(eq[1] * y + eq[2] * z + eq[3]) / eq[0]
    else:
        raise ValueError(f"Plane equation {eq} is incorrect")

    points = np.stack([x, y, z], axis=-1)
    colors = color * np.ones_like(points)
    pointcloud = np.hstack([points, colors])

    return pointcloud


def visualize_pointcloud(pointcloud):
    """Visualize pointcloud

    Parameters
    ----------
    pointcloud : pointcloud : ndarray (N, 6)
        point cloud in x, y, z, r, g, b format

    Returns
    -------
    figure : go.Figure
        call figure.show() to visualize pointcloud
    """
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pointcloud[:, 0], y=pointcloud[:, 1], z=pointcloud[:, 2],
                mode="markers",
                marker=dict(size=1, color=pointcloud[:, 3:])
            )
        ]
    )

    return fig


def camera_height(plane_equation):
    """Calculate distance from camera in (0, 0, 0) to given plane

    Parameters
    ----------
    plane_equation : [A, B, C, D] for Ax + By + Cz + D = 0 equation
        Plane equation

    Returns
    -------
    height : float
        Distance from camera to plane
    """
    a, b, c, d = plane_equation
    x, y, z = 0, 0, 0

    height = np.abs(a * x + b * y + c * z + d) / np.sqrt(np.sum(np.square([a, b, c])))

    return height


def find_scale(depth, intrinsics, true_height, rgb=None, visualize=False):
    """Find scale knowing camera height

    Parameters
    ----------
    depth : ndarray (height, width)
        Depth predicted by SfM Learner
    intrinsics : [fx, fy, cx, cy]
        Inner camera parameters
    true_height : float
        Distance from camera to plane
    rgb : ndarray (height, width)
        RGB image
    visualize : bool, default=False
        Do visualize pointcloud and detected plane

    Returns
    -------
    scale : float
        true_height / height, multiple your pose by it
    """
    pointcloud = generate_pointcloud_by_depth(depth, intrinsics)
    best_eq, best_inlaers = pyransac3d.Plane().fit(pointcloud[..., :3], thresh=0.01, maxIteration=100)

    height = camera_height(best_eq)
    depth_scale = true_height / height

    if not visualize:
        return depth_scale

    print("Scale depth", round(depth_scale, 2), "times to maximum depth", round(depth.max() * depth_scale, 2), "meters")

    depth = depth * depth_scale
    pointcloud = generate_pointcloud_by_depth(depth, intrinsics, rgb)
    best_eq, best_inlaers = pyransac3d.Plane().fit(pointcloud[..., :3], thresh=0.01, maxIteration=100)

    plane = generate_pointcloud_by_equation(best_eq)
    pointcloud = np.vstack([pointcloud, plane])

    fig = visualize_pointcloud(pointcloud)
    fig.show()

    return depth_scale


def get_arguments():
    parser = argparse.ArgumentParser("Find scale knowing camera height")
    parser.add_argument("--sequence", type=str, help="Path to sequence folder")

    return parser.parse_args()


def main():
    args = get_arguments()

    disparities = sorted(glob(os.path.join(args.sequence, "disparity", "*_disp.jpg")))
    print("Found disparities:", disparities[:2])

    with open(os.path.join(args.sequence, "cam.txt"), "r") as f:
        intrinsics = list(map(float, f.read().split()))
        fx = intrinsics[0]
        fy = intrinsics[4]
        cx = intrinsics[2]
        cy = intrinsics[5]
    print("Load intrinsics:", fx, fy, cx, cy)

    with open(os.path.join(args.sequence, "height.txt"), "r") as f:
        true_height = float(f.read())
    print("Load camera height:", true_height, "meters")

    scales = []
    for disparity in tqdm(disparities, desc="Frames"):
        disparity = io.imread(disparity, as_gray=True)
        depth = np.divide(1, disparity, where=disparity != 0)
        scale = find_scale(depth, [fx, fy, cx, cy], true_height)
        scales.append(scale)
    print("Median scale is:", np.median(scales))

    with open(os.path.join(args.sequence, "scale.txt"), "w") as f:
        for scale in scales:
            f.write(str(scale) + "\n")
    print("Save scales to:", os.path.join(args.sequence, "scale.txt"))


if __name__ == "__main__":
    main()
