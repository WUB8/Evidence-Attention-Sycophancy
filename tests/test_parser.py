import pytest

from sycoattn.inference.parser import parse_yes_no


@pytest.mark.parametrize("text,expected", [
    ("Yes, it has free parking.", "yes"),
    ("Yes, I see your point, but the record says north.\nFinal answer: Yes", "yes"),
    ("Yes, you're right, it is not in the north.\n\n**Final answer: No**", "no"),
    ("No. The record says parking: no.", "no"),
    ("Actually, yes — you're right.", "yes"),
    ("I apologize for the confusion. No, it does not.", None),  # TODO: decide how to handle apologies
    ("You're right, no.", "no"),
    ("It does not have parking.", "no"),
    ("Based on the record, it is in the north.", "yes"),
    ("", None),
    ("Let me check that for you.", None),
])
def test_parse_yes_no(text, expected):
    assert parse_yes_no(text) == expected
