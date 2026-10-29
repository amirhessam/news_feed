"""Static checks for the SAM template."""

from pathlib import Path

import yaml

TEMPLATE = Path(__file__).parents[1] / "template.yaml"


class _CloudFormationLoader(yaml.SafeLoader):
    pass


def _tag(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {suffix: value}


_CloudFormationLoader.add_multi_constructor("!", _tag)


def _template() -> dict:
    return yaml.load(TEMPLATE.read_text(encoding="utf-8"), Loader=_CloudFormationLoader)


def _statements(function: dict) -> list[dict]:
    return [statement for policy in function["Properties"]["Policies"] for statement in policy["Statement"]]


def test_template_defines_the_weekly_stack():
    template = _template()
    resources = template["Resources"]
    publish = resources["PublishDigest"]["Properties"]
    serve = resources["ServeFeed"]["Properties"]
    assert publish["Handler"] == "handler.publish_digest"
    assert publish["Timeout"] == 900
    assert publish["MemorySize"] == 1024
    assert publish["Events"]["WeeklySchedule"]["Properties"]["Schedule"] == {"Ref": "ScheduleExpression"}
    assert serve["Handler"] == "handler.serve_feed"
    assert serve["FunctionUrlConfig"]["AuthType"] == "NONE"
    assert template["Globals"]["Function"]["Architectures"] == ["arm64"]
    assert template["Globals"]["Function"]["Runtime"] == "python3.12"
    assert resources["FeedBucket"]["Properties"]["PublicAccessBlockConfiguration"]["BlockPublicPolicy"] is True

    parameters = template["Parameters"]
    assert parameters["ScheduleExpression"]["Default"] == "cron(0 6 ? * MON *)"
    assert parameters["BedrockModelId"]["Default"] == "amazon.nova-lite-v1:0"
    assert parameters["RankModelId"]["Default"] == "amazon.nova-pro-v1:0"
    assert parameters["Categories"]["Default"] == "cs.AI,cs.LG,cs.CL,cs.CV,stat.ML"
    assert parameters["MaxPapers"]["Default"] == 5000
    assert parameters["AnnouncementLagDays"]["Default"] == 3
    variables = publish["Environment"]["Variables"]
    assert variables["RANK_MODEL_ID"] == {"Ref": "RankModelId"}
    assert variables["ANNOUNCE_LAG_DAYS"] == {"Sub": "${AnnouncementLagDays}"}
    assert variables["MAX_PAPERS"] == {"Sub": "${MaxPapers}"}
    assert "VpcConfig" not in TEMPLATE.read_text(encoding="utf-8")


def test_both_functions_can_list_the_bucket_so_missing_keys_return_404():
    resources = _template()["Resources"]
    for name in ("PublishDigest", "ServeFeed"):
        listing = [
            statement
            for statement in _statements(resources[name])
            if "s3:ListBucket" in statement["Action"]
        ]
        assert listing, name
        assert listing[0]["Resource"] == {"GetAtt": "FeedBucket.Arn"}


def test_publisher_permissions_cover_the_models_and_state_keys():
    statements = _statements(_template()["Resources"]["PublishDigest"])
    actions = {action for statement in statements for action in statement["Action"]}
    assert {"comprehend:DetectSentiment", "comprehend:DetectKeyPhrases", "bedrock:InvokeModel"} <= actions
    state = next(statement for statement in statements if statement["Sid"] == "WriteFeedState")
    assert {"Sub": "${FeedBucket.Arn}/state/*"} in state["Resource"]
    assert {"Sub": "${FeedBucket.Arn}/runs/*"} in state["Resource"]


def test_alarms_exist_only_when_an_alert_email_is_set():
    template = _template()
    assert template["Parameters"]["AlertEmail"]["Default"] == ""
    resources = template["Resources"]
    for name in ("AlertTopic", "PublishDigestErrorsAlarm", "FeedTrafficAlarm"):
        assert resources[name]["Condition"] == "HasAlertEmail"
    errors = resources["PublishDigestErrorsAlarm"]["Properties"]
    assert errors["MetricName"] == "Errors"
    assert errors["Dimensions"][0]["Value"] == {"Ref": "PublishDigest"}
    traffic = resources["FeedTrafficAlarm"]["Properties"]
    assert traffic["MetricName"] == "Invocations"
    assert traffic["Dimensions"][0]["Value"] == {"Ref": "ServeFeed"}
