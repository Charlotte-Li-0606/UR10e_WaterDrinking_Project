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
  use_pgi_sim_control:=false \
  > "${VERIFY_TMP}/ur10e_pgi_visual.urdf"
xacro "${DESCRIPTION}" name:=ur ur_type:=ur10e use_pgi_gripper:=true \
  camera_mount_xyz:="0 -0.065 0.020" camera_mount_rpy:="0 0 0" \
  use_pgi_sim_control:=true \
  simulation_controllers:="${PROJECT_ROOT}/config/pgi_140_80_sim_controllers.yaml" \
  > "${VERIFY_TMP}/ur10e_pgi_sim.urdf"

check_urdf "${VERIFY_TMP}/ur10e_default.urdf"
check_urdf "${VERIFY_TMP}/ur10e_pgi_visual.urdf"
check_urdf "${VERIFY_TMP}/ur10e_pgi_sim.urdf"

# Gazebo's URDF-to-SDF conversion catches missing dynamic inertias and invalid
# fixed-joint preservation that check_urdf intentionally does not validate.
gz sdf -p "${VERIFY_TMP}/ur10e_pgi_sim.urdf" \
  > "${VERIFY_TMP}/ur10e_pgi_sim.sdf" \
  2> "${VERIFY_TMP}/gz_sdf.stderr"
if [ -s "${VERIFY_TMP}/gz_sdf.stderr" ]; then
  cat "${VERIFY_TMP}/gz_sdf.stderr" >&2
  exit 1
fi

python3 - \
  "${VERIFY_TMP}/ur10e_default.urdf" \
  "${VERIFY_TMP}/ur10e_pgi_visual.urdf" \
  "${VERIFY_TMP}/ur10e_pgi_sim.urdf" \
  "${VERIFY_TMP}/ur10e_pgi_sim.sdf" <<'PY'
from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET


default_root = ET.parse(sys.argv[1]).getroot()
pgi_root = ET.parse(sys.argv[2]).getroot()
pgi_sim_root = ET.parse(sys.argv[3]).getroot()
pgi_sdf_root = ET.parse(sys.argv[4]).getroot()


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
pgi_sim_joints = named(pgi_sim_root, "joint")
pgi_sim_controls = named(pgi_sim_root, "ros2_control")
default_controls = named(default_root, "ros2_control")
pgi_visual_controls = named(pgi_root, "ros2_control")

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

# Only the Gazebo physics joints receive a 0.5 mm hard-stop guard band. The
# visual model and ros2_control command interface remain exactly 0.040 m/jaw.
for joint_name in ("pgi_left_finger_joint", "pgi_right_finger_joint"):
    sim_limit = pgi_sim_joints[joint_name].find("limit").attrib
    assert math.isclose(float(sim_limit["lower"]), 0.0)
    assert math.isclose(float(sim_limit["upper"]), 0.0405)

mimic = right.find("mimic").attrib
assert mimic == {
    "joint": "pgi_left_finger_joint",
    "multiplier": "1.0",
    "offset": "0.0",
}

for joint in (left, right):
    dynamics = joint.find("dynamics").attrib
    assert float(dynamics["damping"]) > 0.0

assert set(default_controls) == {"ur"}
assert set(pgi_visual_controls) == {"ur"}
assert set(pgi_sim_controls) == {"ur", "pgi_140_80_sim_system"}
pgi_control = pgi_sim_controls["pgi_140_80_sim_system"]
control_joints = {
    joint.attrib["name"]: joint for joint in pgi_control.findall("joint")
}
assert set(control_joints) == {
    "pgi_left_finger_joint",
    "pgi_right_finger_joint",
}
left_control = control_joints["pgi_left_finger_joint"]
right_control = control_joints["pgi_right_finger_joint"]
assert [item.attrib["name"] for item in left_control.findall("command_interface")] == [
    "position"
]
command_limits = {
    param.attrib["name"]: float(param.text)
    for param in left_control.find("command_interface").findall("param")
}
assert command_limits == {"min": 0.0, "max": 0.040}
assert not right_control.findall("command_interface"), (
    "mimic joint must not expose a command interface"
)
for control_joint in (left_control, right_control):
    assert [item.attrib["name"] for item in control_joint.findall("state_interface")] == [
        "position",
        "velocity",
        "effort",
    ]
    initial_value = control_joint.find("state_interface/param")
    assert initial_value.attrib["name"] == "initial_value"
    assert math.isclose(float(initial_value.text), 0.040)

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

physical_links = {
    "pgi_camera_interposer",
    "pgi_mount",
    "pgi_body",
    "pgi_left_finger",
    "pgi_right_finger",
    "d435i_mount",
    "d435i_link",
}
for link_name in physical_links:
    inertial = pgi_links[link_name].find("inertial")
    assert inertial is not None, f"missing Gazebo inertia: {link_name}"
    assert float(inertial.find("mass").attrib["value"]) > 0.0
    inertia = inertial.find("inertia").attrib
    ixx, iyy, izz = (float(inertia[key]) for key in ("ixx", "iyy", "izz"))
    assert min(ixx, iyy, izz) > 0.0
    assert ixx <= iyy + izz and iyy <= ixx + izz and izz <= ixx + iyy

sdf_model = pgi_sdf_root.find("model")
assert sdf_model is not None
sdf_links = {link.attrib["name"] for link in sdf_model.findall("link")}
assert physical_links <= sdf_links, f"PGI links missing after SDF conversion: {physical_links - sdf_links}"
sdf_frames = {frame.attrib["name"] for frame in sdf_model.findall("frame")}
assert {
    "pgi_grasp_center",
    "d435i_color_optical_frame",
    "d435i_depth_optical_frame",
} <= sdf_frames

assert "feeding_cup_link" in default_links
assert "wrist_rgbd_camera_link" in default_links
assert "feeding_cup_link" not in pgi_links
assert "wrist_rgbd_camera_link" not in pgi_links

print(f"default: {len(default_links)} links, {len(default_joints)} joints")
print(f"PGI: {len(pgi_links)} links, {len(pgi_joints)} joints")
print("PGI stroke: 0.040 m/jaw, 0.080 m total; symmetric mimic verified")
print("PGI transform ownership, provisional inertias, and visual/collision geometry verified")
print("PGI Gazebo hard-stop guard and 0.040 m command limit verified")
print("PGI Gazebo SDF conversion and ros2_control master/mimic interfaces verified")
PY
