import argparse

import numpy as np
import pandas as pd
import plotly.express as px


def scale_rel_poses(predictions, scales):
    for i in range(0, len(predictions)):
        for j in range(min(i, predictions.shape[1])):
            if scales[f"{i - j:07}"] > 1e-3:
                predictions[i - j, j, :, -1] /= scales[f"{i - j:07}"]

    return predictions


def rel2abs(predictions):
    # See: https://github.com/ClementPinard/SfmLearner-Pytorch/issues/120

    for i in range(1, len(predictions)):
        r = predictions[i - 1, 1]
        predictions[i] = r[:, :3] @ predictions[i]
        predictions[i, :, :, -1] = predictions[i, :, :, -1] + r[:, -1]

    return predictions[:, 0]


def read_abs_poses(filepath):
    predictions = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            prediction = np.array(list(map(float, line.split()))).reshape(3, 4)
            predictions.append(prediction)

    return np.array(predictions)


def to_path(predictions, type_):
    # The coordinate systems are defined the following way, where directions
    # are informally given from the drivers view, when looking forward onto
    # the road:
    # x: right,   y: down,  z: forward

    path = []
    prev = {"x": 0, "y": 0, "z": 0, "frame": 1, "length": 0, "type": type_}
    length = 0

    for i in range(0, predictions.shape[0]):
        x =  predictions[i, 0, 3]
        z = -predictions[i, 1, 3]
        y =  predictions[i, 2, 3]

        curr = {"x": x, "y": y, "z": z, "frame": prev["frame"] + 1, "length": prev["length"], "type": type_}
        curr["length"] += ((curr["x"] - prev["x"]) ** 2 + (curr["y"] - prev["y"]) ** 2 + (curr["z"] - prev["z"]) ** 2) ** 0.5

        path.append(curr)
        prev = curr

    return pd.DataFrame(path)


def get_arguments():
    parser = argparse.ArgumentParser(description="Plot predicted and true pathes")

    parser.add_argument("--npy", type=str, help=".npy file with predicted poses", required=True)
    parser.add_argument("--txt", type=str, help=".txt file with true poses")
    parser.add_argument("--scale", type=str, help=".npy file with predicted pose scales")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    pathes = []

    pred_poses = np.load(args.npy)
    if args.scale is not None:
        scales = np.load(args.scale, allow_pickle=True).item()
        pred_poses_scaled = scale_rel_poses(pred_poses.copy(), scales)

        pred_poses_scaled = rel2abs(pred_poses_scaled)
        pred_path_scaled = to_path(pred_poses_scaled, type_="pred_scaled")

    pred_poses = rel2abs(pred_poses)
    pred_path = to_path(pred_poses, type_="pred")    
    pathes.append(pred_path)

    if args.txt is not None:
        true_poses = read_abs_poses(args.txt)
        true_path = to_path(true_poses, type_="true")
        pathes.append(true_path)

    if args.scale is not None:
        pathes.append(pred_path_scaled)

    fig = px.scatter_3d(pd.concat(pathes), x="x", y="y", z="z", color="type", hover_data=["length", "frame"])
    fig.update_layout(scene=dict(aspectmode="data"))
    fig.show()

