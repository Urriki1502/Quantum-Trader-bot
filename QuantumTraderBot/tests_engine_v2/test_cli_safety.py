import json

from qt_v2 import build_parser


def test_cli_has_safety_status_but_no_live_submit_command():
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices

    assert "safety-status" in choices
    assert "live" not in choices
    assert "submit" not in choices
