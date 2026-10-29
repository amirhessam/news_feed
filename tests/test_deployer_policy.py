"""Checks for the IAM policy the deploying user needs."""

import json
import re
from pathlib import Path

POLICY = Path(__file__).parents[1] / "deployer-policy.json"


def _statements() -> list[dict]:
    return json.loads(POLICY.read_text(encoding="utf-8"))["Statement"]


def _actions(statement: dict) -> list[str]:
    actions = statement["Action"]
    return [actions] if isinstance(actions, str) else actions


def _resources(statement: dict) -> list[str]:
    resources = statement["Resource"]
    return [resources] if isinstance(resources, str) else resources


def test_pass_role_is_the_only_conditioned_action():
    conditioned = [statement for statement in _statements() if "Condition" in statement]
    assert len(conditioned) == 1
    assert _actions(conditioned[0]) == ["iam:PassRole"]
    assert conditioned[0]["Condition"] == {"StringEquals": {"iam:PassedToService": "lambda.amazonaws.com"}}
    create = [statement for statement in _statements() if "iam:CreateRole" in _actions(statement)]
    assert create and "Condition" not in create[0]


def test_sam_transform_and_managed_stack_are_allowed():
    statements = _statements()
    transform = "arn:aws:cloudformation:us-east-1:aws:transform/Serverless-2016-10-31"
    assert any(
        "cloudformation:CreateChangeSet" in _actions(statement) and transform in _resources(statement)
        for statement in statements
    )
    stacks = next(statement for statement in statements if statement["Sid"] == "CloudFormationStacks")
    assert any("stack/aws-sam-cli-managed-default/" in resource for resource in _resources(stacks))
    assert any("stack/ai-paper-news/" in resource for resource in _resources(stacks))


def test_every_service_in_the_stack_is_covered():
    prefixes = {action.split(":")[0] for statement in _statements() for action in _actions(statement)}
    assert {"cloudformation", "s3", "lambda", "iam", "events", "sns", "cloudwatch", "logs"} <= prefixes


def test_wildcard_resources_are_read_only():
    for statement in _statements():
        if "*" in _resources(statement):
            for action in _actions(statement):
                verb = action.split(":")[1]
                assert verb.startswith(("Describe", "Get", "List", "Validate")), action


def test_policy_fits_the_managed_policy_size_limit():
    assert len(re.sub(r"\s", "", POLICY.read_text(encoding="utf-8"))) <= 6144
