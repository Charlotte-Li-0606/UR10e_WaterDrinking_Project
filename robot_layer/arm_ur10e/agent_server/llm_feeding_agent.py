#!/usr/bin/env python3
"""Real-LLM runner for the single safe ``feed_water`` tool.

The language model is an intent/decomposition layer only.  It can propose one
JSON call to :meth:`FeedingSkillLibrary.feed_water`; it never receives the
UR10e SDK, camera data, joint targets, generic poses, controllers, grippers,
or direct mouth-contact actions.  The local validator is the authoritative
safety boundary before a ROS/MoveIt tool object is created.

``--mock-llm`` remains an explicit offline demonstration path.  Without it,
normal operation always calls the configured OpenAI-compatible Chat
Completions provider.  A real-provider failure exits safely and never falls
back to the mock plan.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "robot_layer").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("Could not locate ur_drinking_project root")


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_layer.arm_ur10e.agent_tools.feeding_tools import FeedingSkillLibrary  # noqa: E402


SAFE_MODE = "high_level_tool_call"
ALLOWED_TOOLS = frozenset({"feed_water"})
FORBIDDEN_TOOLS = frozenset(
    {
        "get_robot_observation",
        "get_latest_mouth_pose",
        "select_active_target",
        "get_active_target_state",
        "search_for_mouth",
        "wait_for_stable_mouth_pose",
        "compute_pre_mouth_target",
        "move_straw_tip_to_pre_mouth",
        "move_straw_tip_to_mouth",
        "move_straw_tip_to_mouth_optional",
        "adjust_cup_vertical",
        "retreat_to_ready",
        "stop_motion_or_hold_position",
        "arbitrary_pose",
        "move_joints",
        "move_to_pose",
        "move_to_position",
        "trajectory",
        "direct_controller_command",
        "controller_command",
        "open_gripper",
        "close_gripper",
        "grasp",
        "attach",
        "detach",
        "direct_mouth_contact",
    }
)


class PlanValidationError(ValueError):
    """Raised before any tool is constructed or robot action is attempted."""


class ProviderConfigurationError(RuntimeError):
    """Raised when the required real LLM call cannot be completed safely."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _target_selection_from_task(task: str) -> str:
    """Return the deterministic mock target selection from user wording."""
    normalized = task.lower()
    if "left" in normalized:
        return "left"
    if "right" in normalized:
        return "right"
    return "center"


def _plan_instructions() -> str:
    """Return the intentionally small model contract for this MVP."""
    return f"""You are a safe UR10e feeding task decomposition planner.
Output JSON only. Return exactly one JSON object and no markdown.

For every feeding request, return exactly this shape:
{{
  "task": "feed_water",
  "mode": "{SAFE_MODE}",
  "steps": [
    {{
      "tool": "feed_water",
      "args": {{
        "target_selection": "center",
        "execute": false,
        "max_search_time_sec": 30.0,
        "allow_direct_mouth_contact": false,
        "allow_vertical_adjust": true
      }}
    }}
  ]
}}

The only allowed tool is feed_water and there must be exactly one step. Map a
requested person to target_selection "left", "center", or "right". The
maximum search time is 30 seconds. Set execute to the supplied CLI execution
permission. Direct mouth contact is forbidden, so
allow_direct_mouth_contact must be false.

You must not command joints, arbitrary poses, trajectories, controllers,
grippers, grasping, attach/detach, direct perception primitives, or direct
mouth contact. feed_water itself performs the fixed safe pipeline: perception,
active mouth search, target selection, PlanningScene safety objects, optional
MoveIt OctoMap occupancy, flange-down pre-mouth planning, and hold. The local
validator is authoritative and can reject or override your plan.
"""


def _mock_plan(task: str, *, request_execute: bool) -> str:
    """Return a fixed JSON tool call without contacting a real LLM."""
    return json.dumps(
        {
            "task": "feed_water",
            "mode": SAFE_MODE,
            "steps": [
                {
                    "tool": "feed_water",
                    "args": {
                        "target_selection": _target_selection_from_task(task),
                        "execute": request_execute,
                        "max_search_time_sec": 30.0,
                        "allow_direct_mouth_contact": False,
                        "allow_vertical_adjust": True,
                    },
                }
            ],
        },
        separators=(",", ":"),
    )


def _chat_messages(task: str, *, request_execute: bool) -> list[dict[str, str]]:
    """Build the same Chat Completions request for SDK and stdlib clients."""
    return [
        {"role": "system", "content": _plan_instructions()},
        {
            "role": "user",
            "content": (
                f"Natural-language task: {task!r}\n"
                f"CLI execution permission: {request_execute}.\n"
                "Generate the safe JSON plan now."
            ),
        },
    ]


def _request_chat_completions_with_stdlib(
    *,
    api_key: str,
    base_url: str | None,
    model: str,
    task: str,
    request_execute: bool,
) -> str:
    """Use the OpenAI-compatible endpoint when this Python lacks the SDK.

    The project launcher normally supplies the SDK from its virtual
    environment.  This compact fallback keeps the documented direct
    ``python3 ...`` command real-LLM capable too, while retaining exactly the
    same Chat Completions payload and never logging credentials.
    """
    endpoint = (base_url or "https://api.openai.com/v1").rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": _chat_messages(task, request_execute=request_execute),
            "response_format": {"type": "json_object"},
            "max_tokens": 400,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise ProviderConfigurationError(
            f"OpenAI-compatible plan request failed with HTTP {exc.code}; no robot tool was created or executed."
        ) from exc
    except urllib.error.URLError as exc:
        raise ProviderConfigurationError(
            "OpenAI-compatible plan request could not reach the provider; no robot tool was created or executed."
        ) from exc
    try:
        response_json = json.loads(body)
        plan_text = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ProviderConfigurationError(
            "Provider returned an invalid Chat Completions response; no robot tool was created or executed."
        ) from exc
    if not isinstance(plan_text, str) or not plan_text.strip():
        raise ProviderConfigurationError("Provider returned no plan text; no robot tool was created or executed.")
    return plan_text


def _request_real_llm_plan(task: str, model: str | None, *, request_execute: bool) -> str:
    """Return a JSON-only high-level tool plan from the configured provider."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ProviderConfigurationError(
            "OPENAI_API_KEY is not set. Configure OPENAI_API_KEY for normal LLM mode, "
            "or pass --mock-llm for the explicit offline demonstration path."
        )
    selected_model = model or os.environ.get("OPENAI_MODEL")
    if not selected_model:
        raise ProviderConfigurationError(
            "Set OPENAI_MODEL or pass --model. Choose a model available to your provider "
            "that can return JSON text."
        )
    base_url = os.environ.get("OPENAI_BASE_URL")
    try:
        from openai import OpenAI
    except ImportError:
        return _request_chat_completions_with_stdlib(
            api_key=api_key,
            base_url=base_url,
            model=selected_model,
            task=task,
            request_execute=request_execute,
        )
    try:
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": 30.0,
            "max_retries": 1,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=selected_model,
            messages=_chat_messages(task, request_execute=request_execute),
            # Widely supported by Chat Completions-compatible providers. The
            # local validator still decides whether the plan is safe.
            response_format={"type": "json_object"},
            max_tokens=400,
        )
    except Exception as exc:
        raise ProviderConfigurationError(
            f"OpenAI-compatible plan request failed: {exc.__class__.__name__}. "
            "No robot tool was created or executed."
        ) from exc
    if not response.choices:
        raise ProviderConfigurationError("Provider returned no completion choices; no robot tool was created or executed.")
    plan_text = response.choices[0].message.content
    if not isinstance(plan_text, str) or not plan_text.strip():
        raise ProviderConfigurationError("Provider returned no plan text; no robot tool was created or executed.")
    return plan_text


def _validate_selection(value: Any) -> str:
    if not isinstance(value, str):
        raise PlanValidationError("feed_water.target_selection must be one of: left, center, right")
    selection = value.strip().lower()
    if selection not in {"left", "center", "right"}:
        raise PlanValidationError("feed_water.target_selection must be one of: left, center, right")
    return selection


def _validate_feed_water_args(args: Mapping[str, Any], *, cli_execute: bool) -> dict[str, Any]:
    """Normalize the only LLM-callable tool without broadening its surface."""
    allowed_args = {
        "target_selection",
        "execute",
        "max_search_time_sec",
        "allow_direct_mouth_contact",
        "allow_vertical_adjust",
    }
    extra = set(args) - allowed_args
    if extra:
        raise PlanValidationError(f"feed_water received unsupported arguments: {', '.join(sorted(extra))}")
    selection = _validate_selection(args.get("target_selection", "center"))
    try:
        max_search_time_sec = float(args.get("max_search_time_sec", 30.0))
    except (TypeError, ValueError) as exc:
        raise PlanValidationError("feed_water.max_search_time_sec must be finite and between 0.1 and 30.0") from exc
    if isinstance(args.get("max_search_time_sec", 30.0), bool) or not math.isfinite(max_search_time_sec):
        raise PlanValidationError("feed_water.max_search_time_sec must be finite and between 0.1 and 30.0")
    if not 0.1 <= max_search_time_sec <= 30.0:
        raise PlanValidationError("feed_water.max_search_time_sec must be finite and between 0.1 and 30.0")
    requested_execute = args.get("execute", False)
    if not isinstance(requested_execute, bool):
        raise PlanValidationError("feed_water.execute must be a boolean")
    direct_contact = args.get("allow_direct_mouth_contact", False)
    if not isinstance(direct_contact, bool):
        raise PlanValidationError("feed_water.allow_direct_mouth_contact must be a boolean")
    if direct_contact:
        raise PlanValidationError("feed_water direct mouth contact is not supported by the current safety-reviewed MVP")
    vertical_adjust = args.get("allow_vertical_adjust", True)
    if not isinstance(vertical_adjust, bool):
        raise PlanValidationError("feed_water.allow_vertical_adjust must be a boolean")
    return {
        "target_selection": selection,
        # CLI permission is the only authority that can enable physical
        # search/pre-mouth motion. --plan-only always forces this false.
        "execute": bool(cli_execute and requested_execute),
        "max_search_time_sec": max_search_time_sec,
        "allow_direct_mouth_contact": False,
        "allow_vertical_adjust": vertical_adjust,
    }


def validate_plan(plan_text: str, *, cli_execute: bool) -> dict[str, Any]:
    """Parse and normalize untrusted JSON before constructing a ROS tool."""
    try:
        raw = json.loads(plan_text)
    except json.JSONDecodeError as exc:
        raise PlanValidationError(f"LLM output is not valid JSON: {exc.msg}") from exc
    if not isinstance(raw, Mapping):
        raise PlanValidationError("LLM plan must be a JSON object")
    if raw.get("task") != "feed_water":
        raise PlanValidationError("LLM plan task must be 'feed_water'")
    if raw.get("mode") != SAFE_MODE:
        raise PlanValidationError(f"LLM plan mode must be {SAFE_MODE!r}")
    steps = raw.get("steps")
    if not isinstance(steps, list) or len(steps) != 1:
        raise PlanValidationError("LLM plan must contain exactly one high-level feed_water step")
    step = steps[0]
    if not isinstance(step, Mapping):
        raise PlanValidationError("Step 0 must be an object")
    tool = step.get("tool")
    if tool in FORBIDDEN_TOOLS:
        raise PlanValidationError(f"Step 0 requests forbidden tool {tool!r}")
    if tool not in ALLOWED_TOOLS:
        raise PlanValidationError(f"Step 0 requests unknown or disallowed tool {tool!r}")
    args = step.get("args", {})
    if not isinstance(args, Mapping):
        raise PlanValidationError("Step 0 args must be an object")
    return {
        "task": "feed_water",
        "mode": SAFE_MODE,
        "steps": [{"tool": "feed_water", "args": _validate_feed_water_args(args, cli_execute=cli_execute)}],
        "cli_execute": bool(cli_execute),
    }


def execute_validated_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Run the one approved high-level pipeline and return its structured result."""
    library = FeedingSkillLibrary()
    try:
        step = plan["steps"][0]
        result = library.feed_water(**step["args"])
        print(
            json.dumps(
                {"event": "step_result", "index": 0, "tool": "feed_water", "result": _jsonable(result)},
                sort_keys=True,
                default=str,
            )
        )
        return _jsonable(result)
    finally:
        library.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Feed water", help="Natural-language task for the planner.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true", help="Generate and validate only (the default).")
    mode.add_argument("--execute", action="store_true", help="Permit the validated feed_water pipeline to send safe motion.")
    parser.add_argument("--mock-llm", action="store_true", help="Use the fixed offline JSON-only demonstration plan.")
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI-compatible model identifier; overrides OPENAI_MODEL and is unused in mock mode.",
    )
    parser.add_argument("--print-plan", action="store_true", help="Print the normalized plan before any tool is created.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cli_execute = bool(args.execute)
    planner_source = "mock" if args.mock_llm else "llm"
    selected_model = "mock" if args.mock_llm else (args.model or os.environ.get("OPENAI_MODEL"))
    print(
        json.dumps(
            {
                "planner_source": planner_source,
                "mock_llm": bool(args.mock_llm),
                "llm_required": not args.mock_llm,
                "mode": "execute" if cli_execute else "plan_only",
                "model": selected_model,
                "base_url_configured": bool(os.environ.get("OPENAI_BASE_URL")),
            },
            sort_keys=True,
        )
    )
    if args.mock_llm and cli_execute:
        print(
            json.dumps(
                {
                    "event": "warning",
                    "planner_source": "mock",
                    "message": "MOCK EXECUTE MODE: no real LLM was used",
                },
                sort_keys=True,
            )
        )
    try:
        if args.mock_llm:
            plan_text = _mock_plan(args.task, request_execute=cli_execute)
        else:
            plan_text = _request_real_llm_plan(args.task, args.model, request_execute=cli_execute)
        plan = validate_plan(plan_text, cli_execute=cli_execute)
    except ProviderConfigurationError as exc:
        # Normal mode is strictly real-LLM-only. Never substitute a mock plan.
        print(json.dumps({"success": False, "stage": "llm_request", "reason": str(exc)}, sort_keys=True))
        return 2
    except PlanValidationError as exc:
        print(json.dumps({"success": False, "stage": "plan_validation", "reason": str(exc)}, sort_keys=True))
        return 2

    if args.mock_llm:
        print(json.dumps({"event": "mock_plan_used", "planner_source": "mock"}, sort_keys=True))
    else:
        print(json.dumps({"event": "llm_plan_received", "planner_source": "llm", "validated": True}, sort_keys=True))

    if args.print_plan:
        print(json.dumps({"event": "validated_plan", "plan": plan}, indent=2, sort_keys=True))
    if cli_execute:
        result = execute_validated_plan(plan)
    else:
        result = {
            "success": True,
            "tool": "feed_water",
            "final_state": "plan_validated",
            "failed_step": None,
            "reason": None,
            "steps": [],
            "note": "Plan-only mode validated the feed_water call and forced execute=false; no feeding tool was run.",
        }
    print(json.dumps({"event": "agent_result", **result}, indent=2, sort_keys=True, default=str))
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
