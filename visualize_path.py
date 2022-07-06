import argparse

import numpy as np
import pandas as pd
import plotly.express as px


def rel2abs(predictions):
    # See: https://github.com/ClementPinard/SfmLearner-Pytorch/issues/120

    for i in range(1, len(predictions)):
        r = predictions[i - 1, 1]
        predictions[i] = r[:, :3] @ predictions[i]
        predictions[i, :, :, -1] = predictions[i, :, :, -1] + r[:, -1]

    return predictions


def read_abs_poses(filepath):
    predictions = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            prediction = np.array(list(map(float, line.split()))).reshape(3, 4)
            predictions.append(prediction)

    return np.array(predictions)[:, np.newaxis]


def to_path(predictions, type_):
    # The coordinate systems are defined the following way, where directions
    # are informally given from the drivers view, when looking forward onto
    # the road:
    # x: right,   y: down,  z: forward

    path = []
    prev = {"x": 0, "y": 0, "z": 0, "length": 0, "type": type_}
    length = 0
    for i in range(predictions.shape[1]):
        x = predictions[0, i, 0, 3]
        y = predictions[0, i, 2, 3]
        z = -predictions[0, i, 1, 3]

        curr = {"x": x, "y": y, "z": z, "length": prev["length"], "type": type_}
        curr["length"] += ((curr["x"] - prev["x"]) ** 2 + (curr["y"] - prev["y"]) ** 2 + (curr["z"] - prev["z"]) ** 2) ** 0.5

        path.append(curr)
        prev = curr

    for i in range(1, predictions.shape[0]):
        x = predictions[i, -1, 0, 3]
        y = predictions[i, -1, 2, 3]
        z = -predictions[i, -1, 1, 3]

        curr = {"x": x, "y": y, "z": z, "length": prev["length"], "type": type_}
        curr["length"] += ((curr["x"] - prev["x"]) ** 2 + (curr["y"] - prev["y"]) ** 2 + (curr["z"] - prev["z"]) ** 2) ** 0.5

        path.append(curr)
        prev = curr

    return pd.DataFrame(path)


def get_arguments():
    parser = argparse.ArgumentParser(description="Plot predicted and true pathes")

    parser.add_argument("--npy", type=str, help=".npy file with predicted poses", required=True)
    parser.add_argument("--txt", type=str, help=".txt file with true poses")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    pred_poses = rel2abs(np.load(args.npy))
    pred_path = to_path(pred_poses, type_="pred")

    if args.txt is not None:
        true_poses = read_abs_poses(args.txt)
        true_path = to_path(true_poses, type_="true")

        df = pd.concat([pred_path, true_path])
    else:
        df = pred_path

    fig = px.scatter_3d(df, x="x", y="y", z="z", color="type", hover_data=["length"])
    fig.update_layout(scene=dict(aspectmode="data"))
    fig.show()

