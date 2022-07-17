import os
import argparse
from glob import glob

import pydeck
import pyransac3d
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def generate_pointcloud(depth, intrinsics, rgb=None, mask=None, color=None):
    fx, fy, cx, cy = intrinsics

    if rgb is None:
        rgb = np.zeros(depth.shape + (3,), dtype=int)
    if mask is None:
        mask = np.ones_like(depth, dtype=int)

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)

    z = depth
    x = z * (c - cx) / fx
    y = z * (r - cy) / fy
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    if color is not None:
        r = color[0] * np.ones_like(z, dtype=int)
        g = color[1] * np.ones_like(z, dtype=int)
        b = color[2] * np.ones_like(z, dtype=int)

    pointcloud = np.dstack((x, y, z, r, g, b))

    return pointcloud[(z > 0) & (mask > 0)]


def visualize_pointcloud(point_cloud):
    df = pd.DataFrame(point_cloud, columns=["x", "y", "z", "r", "g", "b"])
    df[["r", "g", "b"]] = df[["r", "g", "b"]].astype(int)
    target = [df.x.mean(), df.y.mean(), df.z.mean()]

    point_cloud_layer = pydeck.Layer(
        "PointCloudLayer",
        df,
        get_position=["x", "y", "z"],
        get_color=["r", "g", "b"])

    view_state = pydeck.ViewState(target=target, controller=True, zoom=1, rotation_x=-90)
    view = pydeck.View(type="OrbitView", controller=True)

    r = pydeck.Deck(point_cloud_layer, initial_view_state=view_state, views=[view])

    return r


def camera_height(best_eq):
    a, b, c, d = best_eq
    x, y, z = 0, 0, 0

    h = np.abs(a * x + b * y + c * z + d) / np.sqrt(np.square([a, b, c]).sum())

    return h


def generate_pointcloud_for_plane(eq, color=(255, 0, 0), radius=100):
    # eq: Ax+By+Cz+D = 0
    # y = -(Ax+Cz+D)/B

    XX, ZZ = np.meshgrid(np.linspace(-radius, radius, 10),
                         np.linspace(-radius, radius, 10))
    X = XX.ravel()
    Z = ZZ.ravel()
    Y = -(eq[0] * X + eq[2] * Z + eq[3]) / eq[1]
    R = color[0] * np.ones_like(X, dtype=int)
    G = color[1] * np.ones_like(X, dtype=int)
    B = color[2] * np.ones_like(X, dtype=int)

    pointcloud = np.vstack([X, Y, Z, R, G, B]).transpose()

    return pointcloud


def find_scale(depth, intrinsics, true_height, rgb=None, mask=None, visualize=True):
    pointcloud_segmented = generate_pointcloud(depth, intrinsics, rgb, mask, color=(0, 255, 0))

    best_eq, best_inlaers = pyransac3d.Plane().fit(pointcloud_segmented[..., :3], thresh=0.01, maxIteration=100)

    depth_scale = camera_height(best_eq) / true_height

    if not visualize:
        return depth_scale

    print("Camera height is", camera_height(best_eq))
    print("Scale depth", 1 / depth_scale, "times to maximum depth", depth.max() / depth_scale, "meters")

    pointcloud = generate_pointcloud(depth / depth_scale, rgb)
    pointcloud_segmented = generate_pointcloud(depth / depth_scale, rgb, mask, color=(0, 255, 0))
    best_eq, best_inlaers = pyransac3d.Plane().fit(pointcloud_segmented[..., :3], thresh=0.01, maxIteration=100)
    pointcloud_roadway = generate_pointcloud_for_plane(best_eq, color=(255, 0, 0), radius=depth.max() / depth_scale)
    pointcloud_best_inlaers = pointcloud_segmented[best_inlaers]
    pointcloud_best_inlaers[..., 3:] = (0, 0, 255)

    r = visualize_pointcloud(np.vstack([pointcloud, pointcloud_roadway, pointcloud_best_inlaers]))
    r.to_html("temp.html", open_browser=True)

    return depth_scale


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, help="path to sequence")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    disparities = glob(os.path.join(args.sequence, "disparity", "*_disp.jpg"))
    disparities = sorted(disparities, key=lambda path: int(path.split("/")[-1].split("_")[0]))
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
        scale = find_scale(depth, [fx, fy, cx, cy], true_height, rgb=None, mask=None, visualize=False)
        scale = max(1e-3, scale)
        scales.append(1 / scale)
    print("Median scale is:", np.median(scales))

    with open(os.path.join(args.sequence, "scale.txt"), "w") as f:
        for scale in scales:
            f.write(str(scale) + "\n")
    print("Save scales to:", os.path.join(args.sequence, "scale.txt"))

