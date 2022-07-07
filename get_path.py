import argparse


def create_simple_path(corners, directions, meters, frames):
    mats = []

    path = None
    prev = {"x": 0, "y": 0, "z": 0}
    for frame in range(frames):
        s = meters / frames

        if frame in corners:
            path = directions[corners.index(frame)]

        # The coordinate systems are defined the following way, where directions
        # are informally given from the drivers view, when looking forward onto
        # the road:
        # x: right,   y: down,  z: forward
        curr = prev.copy()
        if path == "F":
            curr["z"] += s
        elif path == "L":
            curr["x"] -= s
        elif path == "B":
            curr["z"] -= s
        elif path == "R":
            curr["x"] += s
        elif path == "U":
            curr["y"] -= s
        elif path == "D":
            curr["y"] += s

        mats.append([0, 0, 0, curr["x"],
                     0, 0, 0, curr["y"], 
                     0, 0, 0, curr["z"]])
        prev = curr

    return mats


def get_arguments():
    parser = argparse.ArgumentParser(description="Plot predicted and true pathes")

    parser.add_argument("--corners", type=int, nargs='+', help="Frame ids of corners separated by space, e.g --corners 1 100 200 300 400")
    parser.add_argument("--directions", type=str, help="Directions (Forward, Left, Backward, Right), e.g --directions FLBRF")
    parser.add_argument("--meters", type=float, help="Length of path in meters")
    parser.add_argument("--frames", type=int, help="Length of path in frames")
    parser.add_argument("--txt", type=str, help=".txt file to write poses")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    mats = create_simple_path(args.corners, args.directions, args.meters, args.frames)
    with open(args.txt, "w") as f:
        for mat in mats:
            f.write(str(mat).strip("[]").replace(",", "") + "\n")

