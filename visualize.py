import os
import argparse

import numpy as np
import pandas as pd


def predictions2rel(predictions):
    """Convert output of test_pose.py to relative poses

    Parameters
    ----------
    predictions : ndarray (N, seq_length, 3, 4)
        An array contains (3, 4) pose matrixes. For each row first pose is identity,
        others are relative to the previous.

        P_i_j is (3, 4) pose matrix for frame i relative to the pose for frame j.
        Note that P_i_i is identity pose.

        Array for seq_length = 5:
        [[P_0_0, P_1_0, P_2_1, P_3_2, P_4_3]
         [P_1_1, P_2_1, P_3_2, P_4_3, P_5_4]
         ...
         [P_N-4_N-4, P_N-3_N-4, P_N-2_N-3, P_N-1_N-2, P_N_N-1]
         [0, 0, 0, 0, 0]
         [0, 0, 0, 0, 0]
         [0, 0, 0, 0, 0]
         [0, 0, 0, 0, 0]]

    Returns
    -------
    rel_poses : ndarray (N, 3, 4)
        An array contains (3, 4) pose matrixes from P_0_0 to P_N_N-1.

    References
    ----------
        https://github.com/ClementPinard/SfmLearner-Pytorch/issues/39
        https://github.com/ClementPinard/SfmLearner-Pytorch/issues/82
    """
    seq_length = predictions.shape[1]
    assert np.all(predictions[-seq_length+1:] == 0)

    # P_0_0, P_1_0, P_2_1, ..., P_N_N-1
    rel_poses = np.concatenate([
        predictions[0, 0].reshape(-1, 3, 4),                      # P_0_0
        predictions[:-seq_length+1, 1].reshape(-1, 3, 4),         # P_1_0, P_2_1, ..., P_N-3_N-4
        predictions[-seq_length, 2:].reshape(-1, 3, 4)], axis=0)  # P_N-2_N-3, P_N-1_N-2, P_N_N-1

    assert rel_poses.shape[0] == predictions.shape[0]

    return rel_poses


def scale_rel(rel_poses, scales):
    """Scale relative poses

    Parameters
    ----------
    rel_poses : ndarray (N, 3, 4)
        An array contains (3, 4) pose matrixes from P_0_0 to P_N_N-1.

    scales : ndarray (N)

    Returns
    -------
    rel_poses : ndarray (N, 3, 4)
        An array contains (3, 4) pose matrixes from P_0_0 to P_N_N-1.
    """

    rel_poses[:, 0, 3] *= scales
    rel_poses[:, 1, 3] *= scales
    rel_poses[:, 2, 3] *= scales

    return rel_poses


def rel2abs(rel_poses):
    """Convert output of test_pose.py to absolute poses

    Parameters
    ----------
    rel_poses : ndarray (N, 3, 4)
        An array contains (3, 4) pose matrixes from P_0_0 to P_N_N-1.

    Returns
    -------
    abs_poses : ndarray (N, 3, 4)
        An array contains (3, 4) pose matrixes in world coordinates.

    References
    ----------
        https://github.com/ClementPinard/SfmLearner-Pytorch/issues/120
    """
    abs_poses = []

    prev = rel_poses[0]
    abs_poses.append(prev)

    for curr in rel_poses[1:]:
        curr = prev[:, :3] @ curr  # Rotate
        curr[:, 3] = curr[:, 3] + prev[:, 3]  # Translate

        abs_poses.append(curr)
        prev = curr

    return np.array(abs_poses)


def kitti2abs(path):
    """Read KITTI Odometry ground truth absolute poses

    Parameters
    ----------
    path : str
        Path to KITTI Odometry ground truth poses .txt file.

        Each row contains flatten (3, 4) absolute pose matrix.

    Returns
    -------
    abs_poses : ndarray (N, 3, 4)
        An array contains (3, 4) pose matrixes in world coordinates.
    """
    abs_poses = []
    with open(path, "r") as f:
        for line in f.readlines():
            abs_pose = np.array(list(map(float, line.split()))).reshape(3, 4)
            abs_poses.append(abs_pose)

    return np.array(abs_poses)


def abs2path(abs_poses, label):
    """Read custom absolute poses format

    Parameters
    ----------
    abs_poses : ndarray (N, 3, 4)
        An array contains (3, 4) pose matrixes in world coordinates.

    label : str
        Label for plotly

    Note
    ----
    The KITTI coordinate systems are defined the following way, where directions
    are informally given from the drivers view, when looking forward onto
    the road:
    x: right,   y: down,  z: forward

    Returns
    -------
    abs_poses : pd.DataFrame for px.scatter_3d
        DataFrame contains x, y, z, frame, length and label columns
    """

    path = []

    length = 0
    prev = {"x": 0, "y": 0, "z": 0, "Frame": 0, "Length": length, "Label": label}

    for frame, pose in enumerate(abs_poses):
        translation = pose[:, 3]
        x, y, z = translation[0], translation[2], -translation[1]

        length += np.sqrt(np.sum(np.square(np.stack([x, y, z]) - np.stack([prev["x"], prev["y"], prev["z"]]))))

        curr = {"x": x, "y": y, "z": z, "Frame": frame + 1, "Length": length, "Label": label}

        path.append(curr)
        prev = curr

    return pd.DataFrame(path)


def get_arguments():
    parser = argparse.ArgumentParser("Create dataframe for trajectories visualization")

    parser.add_argument("--dataset", type=str, help="Path to dataset folder")
    parser.add_argument("--sequence", type=str, help="Sequence name")

    return parser.parse_args()


def main():
    args = get_arguments()

    path = os.path.join(args.dataset, "sequences", args.sequence, "predictions.npy")
    rel_poses = predictions2rel(np.load(path))
    pred_path = abs2path(rel2abs(rel_poses.copy()), label="Trajectory predicted by SfM Learner")
    print("Load predicted path: ", path)

    path = os.path.join(args.dataset, "poses", f"{args.sequence}.txt")
    true_path = abs2path(kitti2abs(path), label="Ground Truth Trajectory")
    print("Load ground truth path: ", path)

    path = os.path.join(args.dataset, "sequences", args.sequence, "scale.txt")
    with open(path, "r") as f:
        scales = list(map(float, f.read().split()))
    scaled_rel_poses = scale_rel(rel_poses.copy(), scales)
    scaled_pred_path = abs2path(rel2abs(scaled_rel_poses), label="Proposed Approach")
    print("Load scales: ", path)

    pathes = [pred_path, true_path, scaled_pred_path]

    path = os.path.join(args.dataset, "sequences", args.sequence, "visualize.csv")
    pd.concat(pathes).to_csv(path)
    print("Save dataframe for plotly: ", path)


if __name__ == "__main__":
    main()
