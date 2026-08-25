#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESCRIPTION="${PROJECT_ROOT}/urdf/ur_gz_feeding_markers.urdf.xacro"

source /opt/ros/jazzy/setup.bash
if [ -f /home/dase-hw101/ros2_ws/install/setup.bash ]; then
  source /home/dase-hw101/ros2_ws/install/setup.bash
fi
set -u

VERIFY_TMP="$(mktemp -d)"
trap 'rm -r "${VERIFY_TMP}"' EXIT

xacro "${PROJECT_ROOT}/urdf/pgi_140_80_macro.xacro" \
  > "${VERIFY_TMP}/pgi_macro_only.urdf"
xacro "${DESCRIPTION}" name:=ur ur_type:=ur10e use_pgi_gripper:=false \
  > "${VERIFY_TMP}/ur10e_default.urdf"
xacro "${DESCRIPTION}" name:=ur ur_type:=ur10e use_pgi_gripper:=true \
  camera_mount_xyz:="0 -0.065 0.020" camera_mount_rpy:="0 0 0" \
  > "${VERIFY_TMP}/ur10e_pgi.urdf"

check_urdf "${VERIFY_TMP}/ur10e_default.urdf"
check_urdf "${VERIFY_TMP}/ur10e_pgi.urdf"

python3 - "${VERIFY_TMP}/ur10e_default.urdf" "${VERIFY_TMP}/ur10e_pgi.urdf" <<'PY'
from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET


default_root = ET.parse(sys.argv[1]).getroot()
pgi_root = ET.parse(sys.argv[2]).getroot()


def named(root: ET.Element, tag: str) -> dict[str, ET.Element]:
    result: dict[str, ET.Element] = {}
    for element in root.findall(tag):
        name = element.attrib["name"]
        if name in result:
            raise AssertionError(f"duplicate {tag} name: {name}")
        result[name] = element
    return result


def element_shape(element: ET.Element):
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        tuple(element_shape(child) for child in element),
    )


default_links = named(default_root, "link")
default_joints = named(default_root, "joint")
pgi_links = named(pgi_root, "link")
pgi_joints = named(pgi_root, "joint")

required_links = {
    "pgi_camera_interposer",
    "pgi_mount",
    "pgi_body",
    "pgi_left_finger",
    "pgi_right_finger",
    "pgi_grasp_center",
    "d435i_mount",
    "d435i_link",
    "d435i_color_optical_frame",
    "d435i_depth_optical_frame",
}
missing = required_links - pgi_links.keys()
assert not missing, f"missing PGI links: {sorted(missing)}"
assert not (required_links & default_links.keys()), "PGI links leaked into default description"

left = pgi_joints["pgi_left_finger_joint"]
right = pgi_joints["pgi_right_finger_joint"]
assert left.attrib["type"] == "prismatic"
assert right.attrib["type"] == "prismatic"
assert left.find("axis").attrib["xyz"] == "1 0 0"
assert right.find("axis").attrib["xyz"] == "-1 0 0"

left_limit = left.find("limit").attrib
right_limit = right.find("limit").attrib
for limit in (left_limit, right_limit):
    assert math.isclose(float(limit["lower"]), 0.0)
    assert math.isclose(float(limit["upper"]), 0.040)
assert math.isclose(2.0 * float(left_limit["upper"]), 0.080)

mimic = right.find("mimic").attrib
assert mimic == {
    "joint": "pgi_left_finger_joint",
    "multiplier": "1.0",
    "offset": "0.0",
}

expected_parents = {
    "tool0-pgi_camera_interposer": ("tool0", "pgi_camera_interposer"),
    "pgi_camera_interposer-pgi_mount": ("pgi_camera_interposer", "pgi_mount"),
    "pgi_mount-pgi_body": ("pgi_mount", "pgi_body"),
    "pgi_camera_interposer-d435i_mount": ("pgi_camera_interposer", "d435i_mount"),
}
for joint_name, (parent, child) in expected_parents.items():
    joint = pgi_joints[joint_name]
    assert joint.find("parent").attrib["link"] == parent
    assert joint.find("child").attrib["link"] == child

for link_name in (
    "pgi_camera_interposer",
    "pgi_mount",
    "pgi_body",
    "pgi_left_finger",
    "pgi_right_finger",
    "d435i_mount",
    "d435i_link",
):
    link = pgi_links[link_name]
    visual_origin = link.find("visual/origin").attrib
    collision_origin = link.find("collision/origin").attrib
    assert visual_origin == collision_origin, f"visual/collision origin mismatch: {link_name}"
    visual_geometry = element_shape(link.find("visual/geometry"))
    collision_geometry = element_shape(link.find("collision/geometry"))
    assert visual_geometry == collision_geometry, f"visual/collision geometry mismatch: {link_name}"

body_inertia = pgi_links["pgi_body"].find("inertial/inertia").attrib
ixx, iyy, izz = (float(body_inertia[key]) for key in ("ixx", "iyy", "izz"))
assert min(ixx, iyy, izz) > 0.0
assert ixx <= iyy + izz and iyy <= ixx + izz and izz <= ixx + iyy

assert "feeding_cup_link" in default_links
assert "wrist_rgbd_camera_link" in default_links
assert "feeding_cup_link" not in pgi_links
assert "wrist_rgbd_camera_link" not in pgi_links

print(f"default: {len(default_links)} links, {len(default_joints)} joints")
print(f"PGI: {len(pgi_links)} links, {len(pgi_joints)} joints")
print("PGI stroke: 0.040 m/jaw, 0.080 m total; symmetric mimic verified")
print("PGI transform ownership, inertia, and visual/collision geometry verified")
PY
