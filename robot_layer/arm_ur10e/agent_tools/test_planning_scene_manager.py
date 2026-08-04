"""No-motion checks for deterministic multi-person PlanningScene geometry."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from moveit_msgs.msg import CollisionObject

from robot_layer.arm_ur10e.agent_tools.planning_scene_manager import (
    COMBINED_TOOL_COLLISION_CENTER_TOOL0_M,
    COMBINED_TOOL_COLLISION_LINK_NAME,
    COMBINED_TOOL_COLLISION_OBJECT_ID,
    COMBINED_TOOL_COLLISION_SIZE_M,
    COMBINED_TOOL_COLLISION_TOUCH_LINKS,
    PlanningSceneObstacleConfig,
    PlanningSceneObstacleManager,
    combined_tool_collision_verification,
)


def _manager() -> PlanningSceneObstacleManager:
    manager = PlanningSceneObstacleManager.__new__(PlanningSceneObstacleManager)
    manager.config = PlanningSceneObstacleConfig()
    return manager


class MultiPersonPlanningSceneTest(unittest.TestCase):
    def test_combined_tool_box_is_attached_downward_from_flange_plane(self) -> None:
        attached = PlanningSceneObstacleManager.build_combined_tool_attached_collision()

        self.assertEqual(COMBINED_TOOL_COLLISION_LINK_NAME, attached.link_name)
        self.assertEqual(COMBINED_TOOL_COLLISION_OBJECT_ID, attached.object.id)
        self.assertEqual(
            COMBINED_TOOL_COLLISION_LINK_NAME,
            attached.object.header.frame_id,
        )
        self.assertEqual(
            list(COMBINED_TOOL_COLLISION_TOUCH_LINKS),
            list(attached.touch_links),
        )
        self.assertEqual(
            list(COMBINED_TOOL_COLLISION_SIZE_M),
            list(attached.object.primitives[0].dimensions),
        )
        pose = attached.object.primitive_poses[0]
        self.assertEqual(
            list(COMBINED_TOOL_COLLISION_CENTER_TOOL0_M),
            [pose.position.x, pose.position.y, pose.position.z],
        )
        self.assertAlmostEqual(0.0, pose.position.z - 0.15)
        self.assertAlmostEqual(0.30, pose.position.z + 0.15)
        self.assertTrue(
            combined_tool_collision_verification([attached])["success"]
        )

    def test_apply_combined_tool_box_uses_robot_state_diff_and_verifies(self) -> None:
        manager = _manager()
        attached = manager.build_combined_tool_attached_collision()
        manager._apply_scene = Mock(return_value={"success": True})
        manager._attached_collision_objects = Mock(
            return_value={"success": True, "attached_objects": [attached]}
        )

        result = manager.apply_combined_tool_collision()

        self.assertTrue(result["success"])
        scene = manager._apply_scene.call_args.args[0]
        self.assertTrue(scene.is_diff)
        self.assertTrue(scene.robot_state.is_diff)
        self.assertEqual(
            COMBINED_TOOL_COLLISION_OBJECT_ID,
            scene.robot_state.attached_collision_objects[0].object.id,
        )
        self.assertTrue(
            result["verification"]["real_execution_geometry_complete"]
        )

    def test_combined_tool_verification_rejects_dimension_change(self) -> None:
        attached = PlanningSceneObstacleManager.build_combined_tool_attached_collision()
        attached.object.primitives[0].dimensions[2] = 0.20

        result = combined_tool_collision_verification([attached])

        self.assertFalse(result["success"])
        self.assertFalse(result["checks"]["dimensions"])

    def test_combined_tool_verification_accepts_moveit_pose_canonicalization(self) -> None:
        attached = PlanningSceneObstacleManager.build_combined_tool_attached_collision()
        attached.object.pose.position.z = 0.15
        attached.object.primitive_poses[0].position.z = 0.0

        result = combined_tool_collision_verification([attached])

        self.assertTrue(result["success"])
        self.assertEqual([0.0, 0.0, 0.15], result["actual_center_tool0_m"])

    def test_geometry_is_built_behind_each_face_from_the_camera(self) -> None:
        manager = _manager()
        objects, people = manager.build_collision_objects_for_people(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            camera_position=[0.0, 0.0, 1.0],
        )
        self.assertEqual(6, len(objects))
        self.assertEqual([1.0, 0.0, 0.0], people[0]["behind_face_unit_vector"])
        self.assertEqual([0.0, 1.0, 0.0], people[1]["behind_face_unit_vector"])
        self.assertEqual([1.15, 0.0, 1.03], people[0]["face_safety_center"])
        self.assertEqual("real_human_obstacle_1_torso", objects[4].id)

    def test_apply_people_removes_legacy_and_stale_multi_person_objects(self) -> None:
        manager = _manager()
        expected_ids = {
            "real_human_obstacle_0_head",
            "real_human_obstacle_0_torso",
            "real_human_obstacle_0_face_safety",
        }
        manager._world_object_ids = Mock(
            side_effect=[
                {
                    "success": True,
                    "object_ids": [
                        "human_head_collision",
                        "human_torso_collision",
                        "face_safety_zone",
                        "table_platform_collision",
                        "real_human_obstacle_9_head",
                        "unmanaged_fixture",
                    ],
                },
                {"success": True, "object_ids": sorted(expected_ids | {"unmanaged_fixture"})},
            ]
        )
        manager._apply_scene = Mock(return_value={"success": True})

        result = manager.apply_people(
            [[1.0, 0.0, 1.0]],
            camera_position=[0.0, 0.0, 1.0],
        )

        self.assertTrue(result["success"])
        self.assertEqual(
            [
                "face_safety_zone",
                "human_head_collision",
                "human_torso_collision",
                "real_human_obstacle_9_head",
                "table_platform_collision",
            ],
            result["removed_stale_object_ids"],
        )
        scene = manager._apply_scene.call_args.args[0]
        removed = {
            collision.id
            for collision in scene.world.collision_objects
            if collision.operation == CollisionObject.REMOVE
        }
        self.assertEqual(set(result["removed_stale_object_ids"]), removed)

    def test_remove_cleans_legacy_and_dynamic_multi_person_objects(self) -> None:
        manager = _manager()
        manager._world_object_ids = Mock(
            side_effect=[
                {
                    "success": True,
                    "object_ids": [
                        "human_head_collision",
                        "real_human_obstacle_0_head",
                        "real_human_obstacle_1_torso",
                        "unmanaged_fixture",
                    ],
                },
                {"success": True, "object_ids": ["unmanaged_fixture"]},
            ]
        )
        manager._apply_scene = Mock(return_value={"success": True})

        result = manager.remove()

        self.assertTrue(result["success"])
        self.assertEqual(
            [
                "human_head_collision",
                "real_human_obstacle_0_head",
                "real_human_obstacle_1_torso",
            ],
            result["object_ids"],
        )


if __name__ == "__main__":
    unittest.main()
