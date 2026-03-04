#!/usr/bin/env python3
"""Convert box obstacles from SuperMarket_MapData.json to 2D vertices.

The input JSON has a top level key "Boxs" containing objects with
"position", "rotation" and "scale".  Position is the centre point
(x,y,z).  Rotation is given as Euler angles in degrees (x,y,z) and we
interpret the yaw about the vertical axis (rotation.y, falling back to
rotation.z) as the orientation of the box on the ground plane.  Scale
contains the full lengths in each axis; we only care about the x and z
lengths for the horizontal footprint.

This script generates for each box the four corner vertices on the
x-z plane, ordered counter–clockwise, and writes them as JSON.

Example usage::

    python3 map_trans.py SuperMarket_MapData.json >boxes_2d.json

"""

import argparse
import json
import math
from pathlib import Path


def compute_vertices(box):
    # centre coordinates
    cx = box["position"]["x"]
    cz = box["position"]["z"]

    # extents
    hx = box["scale"]["x"] / 2.0
    hz = box["scale"]["z"] / 2.0

    # yaw angle in degrees around up axis. prefer rotation.y else rotation.z
    rot = box.get("rotation", {})
    yaw_deg = rot.get("y", None)
    if yaw_deg is None:
        yaw_deg = rot.get("z", 0.0)
    yaw = math.radians(yaw_deg)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)

    # corners in local (unrotated) coordinates, CCW order
    local = [
        (-hx, -hz),
        (hx, -hz),
        (hx,  hz),
        (-hx,  hz),
    ]

    world = []
    for lx, lz in local:
        wx = cos_y * lx - sin_y * lz + cx
        wz = sin_y * lx + cos_y * lz + cz
        world.append([wx, wz])
    return world


def convert_from_dict(data):
    """从内存中的地图字典生成 boxes_2d 格式，供 TCP 服务端等直接调用。
    支持 key "boxes"（协议）或 "Boxs"（历史文件）。
    返回: [{"vertices": [[x,z], ...]}, ...]，与 convert() 一致。
    """
    boxes = data.get("boxes", data.get("Boxs", []))
    output = []
    for b in boxes:
        verts = compute_vertices(b)
        output.append({"vertices": verts})
    return output


def convert(input_path):
    with open(input_path, "r") as f:
        data = json.load(f)
    return convert_from_dict(data)


def main():
    parser = argparse.ArgumentParser(
        description="Convert static obstacles to 2D vertex lists.")
    parser.add_argument("input", type=Path, help="Path to JSON map data")
    parser.add_argument("--output", "-o", type=Path,
                        help="Output file (defaults to stdout)")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)
    out = convert_from_dict(data)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
    else:
        json.dump(out, sys.stdout, indent=2)


if __name__ == "__main__":
    import sys
    main()
