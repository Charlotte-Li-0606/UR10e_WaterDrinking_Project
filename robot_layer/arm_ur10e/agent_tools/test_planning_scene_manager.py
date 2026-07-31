"""No-motion checks for deterministic multi-person PlanningScene geometry."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from moveit_msgs.msg import CollisionObject

from robot_layer.arm_ur10e.agent_tools.planning_scene_manager import (
    PlanningSceneObstacleConfig,
    PlanningSceneObstacleManager,
)


def _manager() -> PlanningSceneObstacleManager:
    manager = PlanningSceneObstacleManager.__new__(PlanningSceneObstacleManager)
    manager.config = PlanningSceneObstacleConfig()
    return manager


class MultiPersonPlanningSceneTest(unittest.TestCase):
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
